import time
import threading
import logging

from katpoint import Timestamp


logger = logging.getLogger(__name__)


class Bed(object):
    """A place where one thread sleeps, to be awoken by another thread."""
    def __init__(self):
        self.awake = threading.Event()
        self.time_to_wake = None

    def occupied(self):
        return self.time_to_wake is not None and not self.awake.isSet()

    def climb_in(self, time_to_wake, seconds):
        self.time_to_wake = time_to_wake
        self.awake.wait(seconds)
        self.time_to_wake = None
        self.awake.clear()

    def wake_up(self):
        self.awake.set()


class SingleThreadError(Exception):
    """The SingleThreadLock only allows a single thread ever to use it."""


class SingleThreadLock(object):
    """A lock that only ever allows one thread to use it."""
    def __init__(self):
        self._lock = threading.Lock()
        self.thread_name = ''

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()
        # Don't suppress exceptions
        return False

    def current_thread_name(self):
        thread = threading.current_thread()
        return "%s [%d]" % (thread.name, thread.ident)

    def acquire(self, blocking=True):
        """Acquire lock, raising exception if thread changed."""
        current_thread = self.current_thread_name()
        if self.thread_name and current_thread != self.thread_name:
            msg = 'SingleThreadLock already used by thread %r, cannot accept %r' % \
                  (self.thread_name, current_thread)
            raise SingleThreadError(msg)
        if not self._lock.acquire(False):
            msg = 'Thread %r has re-entered SingleThreadLock' % (self.thread_name,)
            raise SingleThreadError(msg)
        self.thread_name = current_thread
        return True

    def release(self):
        self._lock.release()


class SleepWarpClock(object):
    """Time source with Bed that can warp ahead when both threads sleep."""
    def __init__(self, start_time=None, warp=False):
        self.warp = warp
        self.offset = 0.0 if start_time is None else \
                      Timestamp(start_time).secs - time.time()
        self.bed = Bed()
        self.master_lock = SingleThreadLock()
        self.slave_lock = SingleThreadLock()

    def time(self):
        return time.time() + self.offset

    def check_and_wake_slave(self, timestamp=None):
        timestamp = self.time() if timestamp is None else timestamp
        with self.master_lock:
            if self.bed.occupied() and timestamp >= self.bed.time_to_wake:
                self.bed.wake_up()

    def master_sleep(self, seconds):
        with self.master_lock:
            if self.warp and self.bed.occupied():
                self.offset += seconds
                logger.debug('Master %r warped %g s ahead at %.2f' %
                             (self.master_lock.thread_name, seconds, self.time()))
            else:
                time.sleep(seconds)
                logger.debug('Master %r slept for %g s at %.2f' %
                             (self.master_lock.thread_name, seconds, self.time()))

    def slave_sleep(self, seconds):
        with self.slave_lock:
            logger.debug('Slave %r going to bed for %g s at %.2f' %
                         (self.slave_lock.thread_name, seconds, self.time()))
            self.bed.climb_in(self.time() + seconds, seconds)
            logger.debug('Slave %r woke up at %.2f' %
                         (self.slave_lock.thread_name, self.time(),))

    sleep = slave_sleep


class PeriodicUpdaterThread(threading.Thread):
    """Thread which periodically updates a group of components."""
    def __init__(self, components, clock, period=0.1):
        threading.Thread.__init__(self)
        self.name = 'UpdateThread'
        self.components = components
        self.clock = clock
        self.period = period
        self.last_update = None
        self._thread_active = True

    def __enter__(self):
        """Enter context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context and stop the system."""
        self.stop()
        self.join()
        # Don't suppress exceptions
        return False

    def run(self):
        while self._thread_active:
            timestamp = self.clock.time()
            for component in self.components:
                component.update(timestamp, self.last_update)
            self.last_update = timestamp
            after_update = self.clock.time()
            update_time = after_update - timestamp
            remaining_time = self.period - update_time
            if remaining_time < 0:
                logger.warn("Update thread is struggling: updates take "
                            "%g seconds but repeat every %g seconds" %
                            (update_time, self.period))
            self.clock.check_and_wake_slave(after_update)
            self.clock.master_sleep(remaining_time if remaining_time > 0 else 0)

    def stop(self):
        self._thread_active = False
