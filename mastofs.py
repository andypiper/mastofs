#!/usr/bin/env python3
import argparse
import datetime
import errno
import logging
import operator
import os
import signal
import stat
import sys
import time
from collections import deque
from enum import Enum, auto
from queue import Empty, Queue
from threading import RLock, Thread
from time import sleep

import requests
from cachetools import TTLCache, cachedmethod
from fuse import FUSE, FuseOSError, LoggingMixIn, Operations
from mastodon import Mastodon
from mastodon.errors import (MastodonAPIError, MastodonNetworkError,
                             MastodonRatelimitError)
from mastodon.return_types import (Account, MediaAttachment, Notification,
                                   Status)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('MastoFS')


class PathType(Enum):
    FILE = auto()
    DIRECTORY = auto()
    SYMLINK = auto()


class RateLimitManager:
    """
    Manages API rate limiting
    """

    def __init__(self, max_requests=300, window_seconds=300, initial_backoff=1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.initial_backoff = initial_backoff
        self.request_timestamps = deque()
        self.lock = RLock()
        self.current_backoff = initial_backoff

    def register_request(self):
        """Register that a request was made and clean up old timestamps"""
        current_time = time.time()
        with self.lock:
            # Remove timestamps older than our window
            while (self.request_timestamps and
                   self.request_timestamps[0] < current_time - self.window_seconds):
                self.request_timestamps.popleft()

            # Add current timestamp
            self.request_timestamps.append(current_time)

    def should_throttle(self):
        """Check if we should throttle"""
        with self.lock:
            # Clean up old timestamps
            current_time = time.time()
            while (self.request_timestamps and
                   self.request_timestamps[0] < current_time - self.window_seconds):
                self.request_timestamps.popleft()

            # If we're close to the limit (90%), start throttling
            return len(self.request_timestamps) > (0.9 * self.max_requests)

    def requests_in_window(self):
        """Count requests in current window"""
        with self.lock:
            current_time = time.time()
            while (self.request_timestamps and
                   self.request_timestamps[0] < current_time - self.window_seconds):
                self.request_timestamps.popleft()
            return len(self.request_timestamps)

    def backoff_time(self):
        """Calculate backoff time based on current usage"""
        with self.lock:
            request_ratio = len(self.request_timestamps) / self.max_requests
            # Linear backoff based on ratio
            if request_ratio > 0.95:
                return self.current_backoff * 4
            elif request_ratio > 0.8:
                return self.current_backoff * 2
            elif request_ratio > 0.5:
                return self.current_backoff
            return 0

    def handle_rate_limit_error(self):
        """Increase backoff time when we hit rate limit"""
        with self.lock:
            self.current_backoff = min(
                self.current_backoff * 2, 30)  # Cap at 30 seconds
            return self.current_backoff

    def reset_backoff(self):
        """Reset backoff to initial value"""
        with self.lock:
            if len(self.request_timestamps) < (0.3 * self.max_requests):
                self.current_backoff = self.initial_backoff


class ApiRequestWorker:
    """
    Worker thread to handle API requests in the background
    """

    def __init__(self, rate_limit_manager):
        self.rate_limiter = rate_limit_manager
        self.request_queue = Queue()
        self.result_cache = {}
        self.lock = RLock()
        self.thread = Thread(target=self._worker_thread, daemon=True)
        self.running = True
        self.thread.start()

    def _worker_thread(self):
        """Background thread to process API requests"""
        while self.running:
            try:
                # Get a request to process
                request_id, api_function, args, kwargs = self.request_queue.get(
                    timeout=1)

                # Check if we should throttle
                backoff = self.rate_limiter.backoff_time()
                if backoff > 0:
                    logger.debug(
                        f"Rate limiting: backing off for {backoff:.2f}s")
                    sleep(backoff)

                # Make the API call
                try:
                    self.rate_limiter.register_request()
                    result = api_function(*args, **kwargs)
                    with self.lock:
                        self.result_cache[request_id] = (True, result)
                    # Reset backoff if we're well under limits
                    self.rate_limiter.reset_backoff()
                except MastodonRatelimitError as e:
                    logger.warning(f"Hit rate limit: {e}")
                    backoff = self.rate_limiter.handle_rate_limit_error()
                    logger.info(
                        f"Backing off for {backoff}s due to rate limit")
                    sleep(backoff)
                    # Re-queue the request
                    self.request_queue.put(
                        (request_id, api_function, args, kwargs))
                except (MastodonNetworkError, MastodonAPIError) as e:
                    logger.error(f"API error: {e}")
                    with self.lock:
                        self.result_cache[request_id] = (False, str(e))
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    with self.lock:
                        self.result_cache[request_id] = (False, str(e))

                # Mark task as done
                self.request_queue.task_done()
            except Empty:
                # No requests to process
                pass
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")

    def submit_request(self, api_function, *args, **kwargs):
        """Submit a request to be processed asynchronously"""
        request_id = id(api_function) + sum(id(arg) for arg in args) + \
            sum(id(k) + id(v) for k, v in kwargs.items())

        # Check if we already have the result
        with self.lock:
            if request_id in self.result_cache:
                success, result = self.result_cache[request_id]
                if success:
                    return result
                else:
                    raise Exception(f"Previous request failed: {result}")

        # Queue the request
        self.request_queue.put((request_id, api_function, args, kwargs))

        # Wait for the result
        while True:
            with self.lock:
                if request_id in self.result_cache:
                    success, result = self.result_cache[request_id]
                    if success:
                        return result
                    else:
                        raise Exception(f"Request failed: {result}")
            sleep(0.1)

    def shutdown(self):
        """Stop the worker thread"""
        self.running = False
        self.thread.join(timeout=5)


class PathItem:
    PATH_TYPE_MAP = {
        "file": PathType.FILE,
        "dir": PathType.DIRECTORY,
        "symlink": PathType.SYMLINK
    }

    def __init__(self, path_type, mtime=None, size=0, symlink_target=None, read_fn=None, listdir_fn=None):
        if isinstance(path_type, str):
            self.path_type = self.PATH_TYPE_MAP.get(path_type, PathType.FILE)
        else:
            self.path_type = path_type
        self.mtime = mtime or time.time()
        self.size = size
        self.symlink_target = symlink_target
        self._read_fn = read_fn
        self._listdir_fn = listdir_fn

    def read(self, offset, length):
        if self.path_type != PathType.FILE:
            raise FuseOSError(errno.EISDIR)
        if not self._read_fn:
            return b""
        data = self._read_fn()
        return data[offset: offset + length]

    def listdir(self):
        if self.path_type != PathType.DIRECTORY:
            raise FuseOSError(errno.ENOTDIR)
        if not self._listdir_fn:
            return []
        return self._listdir_fn()


class MastoFS(LoggingMixIn, Operations):
    def __init__(self, url, token, cache_size=100, cache_ttl=5,
                 long_term_cache_size=5000, long_term_ttl=86400,
                 prefetch_timelines=False, mountpoint=None):
        logger.info(f"Initializing MastoFS with API URL: {url}")

        self.api = Mastodon(
            access_token=token,
            api_base_url=url,
        )

        # Store mountpoint for cleanup
        self.mountpoint = mountpoint

        # Setup rate limiting
        self.rate_limiter = RateLimitManager()
        self.api_worker = ApiRequestWorker(self.rate_limiter)

        # Setup caches
        self.mastodon_object_cache = TTLCache(
            maxsize=cache_size, ttl=cache_ttl)
        self.long_term_posts = TTLCache(
            maxsize=long_term_cache_size, ttl=long_term_ttl)
        self.long_term_accounts = TTLCache(
            maxsize=long_term_cache_size, ttl=long_term_ttl)

        self.write_buffers = {}
        self.reblog_post = None
        self.reblog_last_account = ""
        self.reblog_buffer = None  # Initialize reblog_buffer

        self.prefetch_timelines = prefetch_timelines
        self.prefetch_thread = None
        self.running = True

        # Static directories
        self.base_files = {
            '': {
                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                'children': {
                    'posts': {
                        'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                        'children': {
                            'new': {
                                'file': dict(st_mode=(0o644 | stat.S_IFREG), st_nlink=1, st_size=0),
                                'children': None,
                            },
                            'reblogged': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                        },
                    },
                    'accounts': {
                        'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                        'children': {
                            'me': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                        },
                    },
                    'timelines': {
                        'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                        'children': {
                            'home': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                            'local': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                            'federated': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                            'public': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                        },
                    },
                    'notifications': {
                        'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                        'children': {
                            'all': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                            'mention': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                            'favourite': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                            'reblog': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                            'follow': {
                                'file': dict(st_mode=(0o755 | stat.S_IFDIR), st_nlink=2),
                                'children': {}
                            },
                        },
                    },
                },
            },
        }

        # Create a single requests.Session for reuse
        self.session = requests.Session()

        # Start prefetching if enabled
        if self.prefetch_timelines:
            logger.info("Starting background timeline prefetching")
            self._start_prefetching()

    def _start_prefetching(self):
        def prefetch_thread():
            while self.running:
                try:
                    # Prefetch home timeline
                    self._get_timeline_cached('home')
                    if not self.running:
                        break
                    sleep(60)

                    # Prefetch local timeline
                    self._get_timeline_cached('local')
                    if not self.running:
                        break
                    sleep(60)

                    # Prefetch public timelines
                    self._get_timeline_cached('federated')
                    self._get_timeline_cached('public')

                    # Prefetch notifications
                    self._get_notifications_cached('all')
                    if not self.running:
                        break
                    sleep(60)

                    # Check whether to back off
                    if self.rate_limiter.should_throttle():
                        logger.info(
                            "Throttling prefetch due to rate limit concerns")
                        sleep(300)  # 5 minutes
                    else:
                        sleep(120)  # 2 minutes
                except Exception as e:
                    logger.error(f"Error in prefetch thread: {e}")
                    sleep(60) # Wait a bit before retrying

        self.prefetch_thread = Thread(target=prefetch_thread, daemon=True)
        self.prefetch_thread.start()

    @cachedmethod(operator.attrgetter('mastodon_object_cache'))
    def _get_post_cached(self, post_id):
        """Get a post with caching and rate limiting"""
        try:
            logger.debug(f"Fetching post {post_id}")
            post = self.api_worker.submit_request(self.api.status, post_id)
            if post is not None:
                self.long_term_posts[post_id] = True
            return post
        except Exception as e:
            logger.error(f"Error fetching post {post_id}: {str(e)}")
            return None

    @cachedmethod(operator.attrgetter('mastodon_object_cache'))
    def _get_account_cached(self, account_id):
        """Get an account with caching and rate limiting"""
        try:
            logger.debug(f"Fetching account {account_id}")
            acct = self.api_worker.submit_request(self.api.account, account_id)
            if acct is not None:
                self.long_term_accounts[account_id] = True
            return acct
        except Exception as e:
            logger.error(f"Error fetching account {account_id}: {str(e)}")
            return None

    @cachedmethod(operator.attrgetter('mastodon_object_cache'))
    def _get_timeline_cached(self, timeline_type):
        """Get a timeline with caching and rate limiting"""
        try:
            logger.debug(f"Fetching timeline {timeline_type}")
            if timeline_type == 'home':
                return self.api_worker.submit_request(self.api.timeline_home)
            elif timeline_type == 'local':
                return self.api_worker.submit_request(self.api.timeline_local)
            elif timeline_type == 'federated':
                return self.api_worker.submit_request(self.api.timeline_public)
            elif timeline_type == 'public':
                return self.api_worker.submit_request(
                    self.api.timeline_public, local=False)
            return []
        except Exception as e:
            logger.error(f"Error fetching timeline {timeline_type}: {str(e)}")
            return []

    @cachedmethod(operator.attrgetter('mastodon_object_cache'))
    def _get_notifications_cached(self, notification_type=None):
        """Get notifications with caching and rate limiting"""
        try:
            if notification_type == 'all':
                logger.debug("Fetching all notifications")
                return self.api_worker.submit_request(self.api.notifications)
            elif notification_type in ['mention', 'favourite', 'reblog', 'follow']:
                logger.debug(
                    f"Fetching notifications of type {notification_type}")
                try:
                    return self.api_worker.submit_request(self.api.notifications,
                                                          types=[notification_type])
                except TypeError as e:
                    logger.warning(
                        f"API error with 'types' parameter: {str(e)}")
                    # Fallback: fetch all notifications and filter them manually
                    all_notifications = self.api_worker.submit_request(
                        self.api.notifications)
                    return [n for n in all_notifications if n.type == notification_type]
            return []
        except Exception as e:
            logger.error(
                f"Error fetching notifications of type {notification_type}: {str(e)}")
            return []

    def _get_base_file(self, path):
        if path == '/':
            return self.base_files['']
        curr = self.base_files['']
        parts = path.strip('/').split('/')
        for p in parts:
            kids = curr.get('children', {})
            if p in kids:
                curr = kids[p]
            else:
                return None
        return curr

    def _extract_time(self, obj):
        t = None
        if isinstance(obj, dict) and "created_at" in obj:
            t = obj["created_at"]
        elif hasattr(obj, "created_at"):
            t = getattr(obj, "created_at")
        if not t:
            return time.time()
        if isinstance(t, str):
            try:
                dt = datetime.datetime.fromisoformat(t)
                return dt.timestamp()
            except Exception as e:
                logger.error(f"Error parsing time string: {str(e)}")
                return time.time()
        if isinstance(t, datetime.datetime):
            return t.timestamp()
        return time.time()

    def _fetch_media(self, url):
        """Fetch media content with caching"""
        try:
            # Use the shared session for connection pooling
            with self.session.get(url, timeout=(3.0, 30.0),
                                  stream=True,  # Stream to handle large files better
                                  headers={'User-Agent': 'MastoFS/0.1'}) as r:
                r.raise_for_status()

                # Check content length
                content_length = r.headers.get('Content-Length')
                if content_length and int(content_length) > 10 * 1024 * 1024:  # >10MB
                    logger.warning(
                        f"Large media file detected: {url} ({int(content_length)/1024/1024:.1f} MB)")

                # Stream and read in chunks for better memory handling
                chunks = []
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)

                return b''.join(chunks)
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching media from {url}")
            return b""
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error fetching media from {url}")
            return b""
        except Exception as e:
            logger.error(f"Error fetching media from {url}: {str(e)}")
            return b""

    def _list_keys(self, obj):
        if isinstance(obj, list):
            return [str(i) for i in range(len(obj))]
        if isinstance(obj, dict):
            base_keys = list(obj.keys())
            if isinstance(obj, MediaAttachment):
                if "url" in obj and "file" not in base_keys:
                    base_keys.append("file")
                if "preview_url" in obj and "preview_file" not in base_keys:
                    base_keys.append("preview_file")
            return base_keys
        return []

    def _get_child(self, obj, key):
        if isinstance(obj, list):
            try:
                i = int(key)
                return obj[i]
            except (ValueError, IndexError) as e:
                logger.error(f"Error accessing list item {key}: {str(e)}")
                raise KeyError(f"Invalid list index: {key}")
        if isinstance(obj, dict):
            if isinstance(obj, MediaAttachment):
                if key == "file" and "url" in obj:
                    return self._fetch_media(obj["url"])
                if key == "preview_file" and "preview_url" in obj:
                    return self._fetch_media(obj["preview_url"])
            try:
                return obj[key]
            except KeyError:
                logger.error(f"Key {key} not found in object")
                raise
        raise KeyError("No children")

    def _traverse(self, obj, parts):
        if not parts:
            return obj
        return self._traverse(self._get_child(obj, parts[0]), parts[1:])

    def _resolve_path(self, path):
        # Normalize the path while preserving the leading slash
        norm_path = os.path.normpath(path)
        if path.startswith('/') and not norm_path.startswith('/'):
            norm_path = '/' + norm_path

        # Initialize base_children to None
        base_children = None

        # Handle root directory
        if base_children is not None:
            return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=lambda: base_children)

        # Static base tree
        base_item = self._get_base_file(path)
        if base_item is not None:
            if base_item["children"] is None:
                return PathItem(PathType.FILE, time.time(), 0, read_fn=lambda: b"")
            else:
                base_children = list(base_item["children"].keys())

        # Handle dynamic paths (e.g., /posts, /accounts, /timelines, /notifications)
        parts = norm_path.strip('/').split('/')
        for part in parts:
            if part.startswith('.'):
                raise FuseOSError(errno.ENOENT)

        if parts and parts[0] == 'posts':
            return self._resolve_posts(path, parts)
        if parts and parts[0] == 'accounts':
            return self._resolve_accounts(path, parts)
        if parts and parts[0] == 'timelines':
            return self._resolve_timelines(path, parts)
        if parts and parts[0] == 'notifications':
            return self._resolve_notifications(path, parts)

        # Handle root directory again after base_children is set
        if base_children is not None:
            return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=lambda: base_children)

        # If no match, raise an error
        raise FuseOSError(errno.ENOENT)

    def _resolve_posts(self, path, parts):
        if len(parts) == 1:
            # /posts
            node = self._get_base_file(path)

            # Get the list of posts from the long term cache
            def list_posts():
                base = list(node['children'].keys()
                            ) if node and 'children' in node else []
                for pid in self.long_term_posts:
                    if pid not in base:
                        base.append(pid)
                return base
            return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=list_posts)

        if parts[1] == 'reblogged':
            # /posts/reblogged
            if len(parts) == 2:
                node = self._get_base_file(path)
                if node:
                    return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=lambda: list(node.get('children', {}).keys()))
                return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=lambda: [])
            raise FuseOSError(errno.ENOENT)

        # /posts/<id>/...
        post_id = parts[1]
        if post_id.startswith('.'):
            raise FuseOSError(errno.ENOENT)
        post_obj = self._get_post_cached(post_id)
        if not post_obj:
            raise FuseOSError(errno.ENOENT)

        if len(parts) == 2:
            # Post directory
            t = self._extract_time(post_obj)

            def listdir_post():
                return self._list_keys(post_obj)
            return PathItem(PathType.DIRECTORY, t, listdir_fn=listdir_post)

        # Traverse deeper into the object
        try:
            final_obj = self._traverse(post_obj, parts[2:])
        except (KeyError, IndexError) as e:
            logger.error(f"Error traversing post object: {str(e)}")
            raise FuseOSError(errno.ENOENT)
        return self._make_item_for_obj(final_obj)

    def _resolve_accounts(self, path, parts):
        # /accounts
        if len(parts) == 1:
            # top-level directory => static + known account IDs
            node = self._get_base_file(path)

            def list_accounts():
                base = list(node['children'].keys()
                            ) if node and 'children' in node else []
                # add known from long_term
                for aid in self.long_term_accounts:
                    if aid not in base:
                        base.append(aid)
                return base
            return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=list_accounts)

        # /accounts/me
        if parts[1] == 'me':
            if parts[1].startswith('.'):
                raise FuseOSError(errno.ENOENT)
            try:
                acct = self.api.account_verify_credentials()
                if acct:
                    self.long_term_accounts[str(acct.id)] = True
            except Exception as e:
                logger.error(f"Error verifying account credentials: {str(e)}")
                raise FuseOSError(errno.ENOENT)

            if len(parts) == 2:
                t = self._extract_time(acct)

                def list_me():
                    return self._list_keys(acct)
                return PathItem(PathType.DIRECTORY, t, listdir_fn=list_me)
            try:
                sub = self._traverse(acct, parts[2:])
            except (KeyError, IndexError) as e:
                logger.error(f"Error traversing account object: {str(e)}")
                raise FuseOSError(errno.ENOENT)
            return self._make_item_for_obj(sub)

        # /accounts/<id>/...
        account_id = parts[1]
        acct_obj = self._get_account_cached(account_id)
        if not acct_obj:
            raise FuseOSError(errno.ENOENT)

        if len(parts) == 2:
            # Account directory
            t = self._extract_time(acct_obj)

            def list_acct():
                return self._list_keys(acct_obj)
            return PathItem(PathType.DIRECTORY, t, listdir_fn=list_acct)

        # Traverse deeper into the object
        try:
            sub = self._traverse(acct_obj, parts[2:])
        except (KeyError, IndexError) as e:
            logger.error(f"Error traversing account object: {str(e)}")
            raise FuseOSError(errno.ENOENT)
        return self._make_item_for_obj(sub)

    def _resolve_timelines(self, path, parts):
        # /timelines
        if len(parts) == 1:
            node = self._get_base_file(path)

            def list_top():
                return list(node['children'].keys()) if node and 'children' in node else []
            return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=list_top)
        if len(parts) == 2:
            node = self._get_base_file(path)

            def list_tl():
                base = list(node['children'].keys()
                            ) if node and 'children' in node else []
                posts = self._get_timeline_cached(parts[1])
                base.extend(str(i) for i in range(len(posts)))
                return base
            return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=list_tl)
        if len(parts) == 3:
            timeline_type = parts[1]
            idx_s = parts[2]

            if idx_s.startswith('.'):
                raise FuseOSError(errno.ENOENT)

            try:
                idx = int(idx_s)
            except ValueError as e:
                logger.error(f"Error converting index to integer: {str(e)}")
                raise FuseOSError(errno.ENOENT)

            posts = self._get_timeline_cached(timeline_type)
            if idx < 0 or idx >= len(posts):
                logger.error(
                    f"Timeline index out of range: {idx} for timeline of length {len(posts)}")
                raise FuseOSError(errno.ENOENT)

            post = posts[idx]

            def symlink_target():
                return f"/posts/{post.id}"
            t = self._extract_time(post)
            return PathItem(PathType.SYMLINK, t, symlink_target=symlink_target)
        raise FuseOSError(errno.ENOENT)

    def _resolve_notifications(self, path, parts):
        """Handle /notifications/... paths"""
        # /notifications
        if len(parts) == 1:
            node = self._get_base_file(path)

            def list_notif_types():
                return list(node['children'].keys()) if node and 'children' in node else []
            return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=list_notif_types)

        # /notifications/<type> (all, mention, favourite, reblog, follow)
        if len(parts) == 2:
            notification_type = parts[1]
            if notification_type not in ['all', 'mention', 'favourite', 'reblog', 'follow']:
                logger.error(f"Invalid notification type: {notification_type}")
                raise FuseOSError(errno.ENOENT)

            node = self._get_base_file(path)

            def list_notifications():
                base = list(node['children'].keys()
                            ) if node and 'children' in node else []
                notifications = self._get_notifications_cached(
                    notification_type)
                base.extend(str(i) for i in range(len(notifications)))
                return base
            return PathItem(PathType.DIRECTORY, time.time(), listdir_fn=list_notifications)

        # /notifications/<type>/<index>
        if len(parts) == 3:
            notification_type = parts[1]
            idx_s = parts[2]

            if idx_s.startswith('.'):
                raise FuseOSError(errno.ENOENT)

            try:
                idx = int(idx_s)
            except ValueError as e:
                logger.error(f"Error converting index to integer: {str(e)}")
                raise FuseOSError(errno.ENOENT)

            notifications = self._get_notifications_cached(notification_type)
            if idx < 0 or idx >= len(notifications):
                logger.error(
                    f"Notification index out of range: {idx} for notifications of length {len(notifications)}")
                raise FuseOSError(errno.ENOENT)

            notification = notifications[idx]

            # Handle different notification types appropriately
            t = self._extract_time(notification)

            # Create a safe way to check if attributes exist
            def has_attr(obj, attr):
                return hasattr(obj, attr) and getattr(obj, attr) is not None

            # For notifications with a status reference
            if has_attr(notification, 'status') and has_attr(notification.status, 'id'):
                def symlink_target():
                    return f"/posts/{notification.status.id}"
                return PathItem(PathType.SYMLINK, t, symlink_target=symlink_target)

            # For follow notifications (or any with only an account reference)
            elif notification.type == 'follow' or (has_attr(notification, 'account') and not has_attr(notification, 'status')):
                def symlink_target():
                    return f"/accounts/{notification.account.id}"
                return PathItem(PathType.SYMLINK, t, symlink_target=symlink_target)

            # For all other notification types, create a directory with notification details
            else:
                def listdir_fn():
                    return self._list_keys(notification)
                return PathItem(PathType.DIRECTORY, t, listdir_fn=listdir_fn)

        raise FuseOSError(errno.ENOENT)

    def _make_item_for_obj(self, obj):
        # Turn the object into a PathItem
        if isinstance(obj, Account):
            def symlink_target():
                return f"/accounts/{obj.id}"
            return PathItem(PathType.SYMLINK, self._extract_time(obj), symlink_target=symlink_target)

        if isinstance(obj, Status):
            def symlink_target():
                return f"/posts/{obj.id}"
            return PathItem(PathType.SYMLINK, self._extract_time(obj), symlink_target=symlink_target)

        if isinstance(obj, Notification):
            t = self._extract_time(obj)

            # Create a safe way to check if attributes exist
            def has_attr(obj, attr):
                return hasattr(obj, attr) and getattr(obj, attr) is not None

            # For notifications with a status reference
            if has_attr(obj, 'status') and has_attr(obj.status, 'id'):
                def symlink_target():
                    return f"/posts/{obj.status.id}"
                return PathItem(PathType.SYMLINK, t, symlink_target=symlink_target)

            # For follow notifications (or any with only an account reference)
            elif obj.type == 'follow' or (has_attr(obj, 'account') and not has_attr(obj, 'status')):
                def symlink_target():
                    return f"/accounts/{obj.account.id}"
                return PathItem(PathType.SYMLINK, t, symlink_target=symlink_target)

            # For all other notification types, create a directory with notification details
            else:
                def listdir_fn():
                    return self._list_keys(obj)
                return PathItem(PathType.DIRECTORY, t, listdir_fn=listdir_fn)

        if isinstance(obj, (dict, list)):
            t = self._extract_time(obj)

            def listdir_fn():
                return self._list_keys(obj)
            return PathItem(PathType.DIRECTORY, t, listdir_fn=listdir_fn)

        if isinstance(obj, bytes):
            data_bytes = obj
            size = len(data_bytes)
            t = time.time()
            return PathItem(PathType.FILE, t, size=size, read_fn=lambda: data_bytes)

        # If nothing else: treat as text
        data_str = str(obj).encode('utf-8')
        size = len(data_str)
        t = time.time()
        return PathItem(PathType.FILE, t, size=size, read_fn=lambda: data_str)

    def getattr(self, path, fh=None):
        # We special case the /posts/reblogged directory to allow basically everything without error messages
        # We drop pretty much all writes though
        if path.startswith('/posts/reblogged'):
            parts = path.split('/')
            if len(parts) <= 4 or parts[-1] in ["mentions", "media_attachments", "emojis", "tags", "filtered", "application"]:
                st = {
                    'st_mode': (0o755 | stat.S_IFDIR),
                    'st_nlink': 2,
                    'st_size': 0,
                    'st_mtime': time.time(),
                    'st_atime': time.time(),
                    'st_ctime': time.time()
                }
            else:
                if parts[-1] == "account":
                    if path != self.reblog_last_account:
                        raise FuseOSError(errno.ENOENT)
                    else:
                        st = {
                            'st_mode': (0o755 | stat.S_IFLNK),
                            'st_nlink': 1,
                            'st_size': 0,
                            'st_mtime': time.time(),
                            'st_atime': time.time(),
                            'st_ctime': time.time()
                        }
                        return st
                else:
                    st = {
                        'st_mode': (0o644 | stat.S_IFREG),
                        'st_nlink': 1,
                        'st_size': 0,
                        'st_mtime': time.time(),
                        'st_atime': time.time(),
                        'st_ctime': time.time()
                    }
            return st

        # Otherwise, just resolve and return correct attrs
        item = self._resolve_path(path)
        mode = 0
        if item.path_type == PathType.DIRECTORY:
            mode = (0o755 | stat.S_IFDIR)
        elif item.path_type == PathType.FILE:
            mode = (0o644 | stat.S_IFREG)
        elif item.path_type == PathType.SYMLINK:
            mode = (0o755 | stat.S_IFLNK)

        st = {
            'st_mode': mode,
            'st_nlink': 1,
            'st_size': item.size,
            'st_mtime': item.mtime,
            'st_atime': item.mtime,
            'st_ctime': item.mtime
        }

        return st

    def readdir(self, path, fh):
        # Return the list of files in a directory
        item = self._resolve_path(path)
        if item.path_type != PathType.DIRECTORY:
            raise FuseOSError(errno.ENOTDIR)
        entries = ['.', '..']
        entries.extend(item.listdir())
        return entries

    def readlink(self, path):
        # Return the target of the symlink
        item = self._resolve_path(path)
        if item.path_type != PathType.SYMLINK:
            raise FuseOSError(errno.EINVAL)
        target = item.symlink_target
        if callable(target):
            target = target()
        return os.path.relpath(target, os.path.dirname(path))

    def read(self, path, size, offset, fh):
        # Return contents of the attribute (or potentially downloaded media file)
        item = self._resolve_path(path)
        if item.path_type != PathType.FILE:
            raise FuseOSError(errno.EISDIR)
        return item.read(offset, size)

    def write(self, path, data, offset, fh):
        # Write to /posts/new posts status
        if path == '/posts/new':
            if path not in self.write_buffers:
                self.write_buffers[path] = b""
            self.write_buffers[path] += data
            return len(data)

        # Write to /posts/reblogged/<anything>/id stores the id to reblog on close
        if path.startswith('/posts/reblogged'):
            if len(path.split('/')) == 5:
                if path.split('/')[-1] == "id":
                    self.reblog_buffer = data.decode('utf-8')
            return len(data)

        raise FuseOSError(errno.EROFS)

    def create(self, path, mode, fi=None):
        # required for writing to work correctly in some cases
        if path == '/posts/new':
            self.write_buffers[path] = b""
            return 0

        # Better strategy: allow creation in general, then reblog when writing an id to the appropriate place
        if path.startswith('/posts/reblogged'):
            return 0

        # no creation allowed generally
        raise FuseOSError(errno.EROFS)

    # Have to allow directory creation under /posts/reblogged
    def mkdir(self, path, mode):
        if path.startswith('/posts/reblogged'):
            return 0
        raise FuseOSError(errno.EROFS)

    # Also have to allow file deletion under /posts/reblogged
    # I'm not sure why but symlink copy breaks otherwise??
    def unlink(self, path):
        if path.startswith('/posts/reblogged'):
            return 0
        raise FuseOSError(errno.EROFS)

    def release(self, path, fh):
        # on close: post whatever is in the write buffer
        if path == '/posts/new':
            buf = self.write_buffers.get(path, b"")
            if buf:
                text = buf.decode('utf-8')
                try:
                    # Use the api_worker for rate-limited API access
                    newp = self.api_worker.submit_request(
                        self.api.status_post, text)
                    if newp:
                        self.long_term_posts[str(newp.id)] = True
                        logger.info(
                            f"Successfully posted status with ID: {newp.id}")
                except Exception as e:
                    logger.error(f"Error posting status: {str(e)}")
                finally:
                    if path in self.write_buffers:
                        del self.write_buffers[path]

        # on closing a reblogged post, reblog it
        if path.startswith('/posts/reblogged/'):
            if len(path.split('/')) == 5:
                if path.split('/')[-1] == "id":
                    try:
                        if self.reblog_buffer:
                            # Use the api_worker for rate-limited API access
                            self.api_worker.submit_request(
                                self.api.status_reblog, self.reblog_buffer)
                            logger.info(
                                f"Successfully reblogged status: {self.reblog_buffer}")
                            self.reblog_buffer = None
                        else:
                            logger.warning(
                                "Attempted to reblog with empty buffer")
                    except Exception as e:
                        logger.error(f"Error reblogging status: {str(e)}")
        return 0

    def truncate(self, path, length, fh=None):
        # required for writing to work correctly in many cases
        if path == '/posts/new':
            self.write_buffers[path] = b""
            return 0
        if path.startswith('/posts/reblogged'):
            return 0
        raise FuseOSError(errno.EROFS)

    def symlink(self, target, source):
        if target.startswith('/posts/reblogged'):
            self.reblog_last_account = target
            return 0
        raise FuseOSError(errno.EROFS)

    def shutdown(self):
        """
        Clean up resources
        """
        logger.info("Shutting down MastoFS...")

        # Stop background threads
        self.running = False

        # Flush any pending write operations
        self._flush_write_buffers()

        # Shutdown the API worker
        if hasattr(self, 'api_worker') and self.api_worker:
            logger.debug("Shutting down API worker...")
            self.api_worker.shutdown()

        # Close the shared session
        if hasattr(self, 'session') and self.session:
            logger.debug("Closing requests session...")
            self.session.close()

        # Wait for prefetch thread to terminate if it exists
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            logger.debug("Waiting for prefetch thread to terminate...")
            self.prefetch_thread.join(timeout=2)
            if self.prefetch_thread.is_alive():
                logger.warning("Prefetch thread did not terminate gracefully")

        logger.info("MastoFS shutdown complete")

    def _flush_write_buffers(self):
        """Flush any pending write operations"""
        if not self.write_buffers:
            return

        logger.info(
            f"Flushing {len(self.write_buffers)} pending write buffers...")
        for path, buf in list(self.write_buffers.items()):
            if path == '/posts/new' and buf:
                logger.info("Flushing pending post...")
                try:
                    text = buf.decode('utf-8')
                    if text.strip():  # Only post if there's actual content
                        logger.info(f"Posting pending content: {text[:50]}...")
                        newp = self.api.status_post(text)
                        if newp:
                            logger.info(
                                f"Successfully posted status with ID: {newp.id}")
                except Exception as e:
                    logger.error(f"Error posting pending status: {str(e)}")
            # Remove the buffer regardless of success
            del self.write_buffers[path]

    def __del__(self):
        """Destructor to ensure cleanup happens"""
        try:
            self.shutdown()
        except Exception:
            # Can't log here as logger might be gone
            pass


def main():
    """Main entry point with improved CLI handling and clean shutdown support"""
    parser = argparse.ArgumentParser(
        description='Mount Mastodon as a filesystem using FUSE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'mountpoint', help='Directory where the filesystem will be mounted')
    parser.add_argument(
        'url', help='Mastodon instance URL (e.g., https://mastodon.social)')
    parser.add_argument('token', help='Mastodon access token')

    # Additional options
    parser.add_argument('--cache-size', type=int, default=100,
                        help='Size of the short-term object cache')
    parser.add_argument('--cache-ttl', type=int, default=5,
                        help='TTL in seconds for short-term object cache')
    parser.add_argument('--long-term-cache-size', type=int, default=5000,
                        help='Size of the long-term post/account cache')
    parser.add_argument('--long-term-ttl', type=int, default=86400,
                        help='TTL in seconds for long-term cache (default: 1 day)')
    parser.add_argument('--prefetch', action='store_true',
                        help='Enable background prefetching of timelines')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress all log messages except errors')

    args = parser.parse_args()

    # Configure logging based on arguments
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    # Check if mountpoint exists and is a directory
    if not os.path.exists(args.mountpoint):
        logger.info(
            f"Mountpoint {args.mountpoint} does not exist, creating it...")
        try:
            os.makedirs(args.mountpoint)
        except Exception as e:
            logger.error(f"Failed to create mountpoint directory: {e}")
            sys.exit(1)
    elif not os.path.isdir(args.mountpoint):
        logger.error(
            f"Mountpoint {args.mountpoint} exists but is not a directory")
        sys.exit(1)

    # Create the filesystem
    fs = MastoFS(
        url=args.url,
        token=args.token,
        cache_size=args.cache_size,
        cache_ttl=args.cache_ttl,
        long_term_cache_size=args.long_term_cache_size,
        long_term_ttl=args.long_term_ttl,
        prefetch_timelines=args.prefetch,
        mountpoint=args.mountpoint
    )

    # Handle signals to gracefully unmount
    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        fs.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Mount the filesystem
    try:
        logger.info(f"Mounting MastoFS at {args.mountpoint}...")
        FUSE(fs, args.mountpoint, nothreads=True, foreground=True)
        logger.info(f"MastoFS unmounted from {args.mountpoint}")
    except Exception as e:
        logger.error(f"Failed to mount MastoFS: {e}")
        fs.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()
