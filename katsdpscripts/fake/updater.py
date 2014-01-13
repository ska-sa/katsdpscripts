import time
import threading
import logging

from katpoint import Timestamp


logger = logging.getLogger(__name__)


class WarpClock(object):
    """Time source that can warp ahead during a sleep phase."""
    def __init__(self, start_time=None):
        self.offset = 0.0 if start_time is None else \
                      Timestamp(start_time).secs - time.time()

    def time(self):
        return time.time() + self.offset

    def sleep(self, seconds, warp=False):
        if warp:
            self.offset += seconds
        else:
            time.sleep(seconds)


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


class PeriodicUpdaterThread(threading.Thread):
    """Thread which periodically updates a group of components."""
    def __init__(self, components, dry_run=False, start_time=None, period=0.1):
        threading.Thread.__init__(self)
        self.name = 'UpdateThread'
        self.components = components
        self.dry_run = dry_run
        self.period = period
        self.clock = WarpClock(start_time)
        self.bed = Bed()
        self.last_update = None
        self._thread_active = True

    def run(self):
        while self._thread_active:
            timestamp = self.clock.time()
            for component in self.components:
                component.update(timestamp)
            self.last_update = timestamp
            after_update = self.clock.time()
            update_time = after_update - timestamp
            remaining_time = self.period - update_time
            if remaining_time < 0:
                logger.warn("Update thread is struggling: updates take "
                            "%g seconds but repeat every %g seconds" %
                            (update_time, self.period))
            warp = False
            if self.bed.occupied():
                if after_update >= self.bed.time_to_wake:
                    self.bed.wake_up()
                elif self.dry_run:
                    warp = True
            logger.debug('Updater sleeping for %g s, %s' %
                         (remaining_time, 'warp' if warp else 'normal'))
            self.clock.sleep(remaining_time if remaining_time > 0 else 0, warp)

    def stop(self):
        self._thread_active = False

    def time(self):
        """Current time in UTC seconds since Unix epoch."""
        return self.clock.time()

    def sleep(self, seconds):
        """Sleep for the requested duration in seconds."""
        self.bed.climb_in(self.clock.time() + seconds, seconds)
