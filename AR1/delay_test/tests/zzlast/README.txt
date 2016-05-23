zzlast/
    This folder is reserved for tests that, due to the way they test or exercise the
    system, should run at the end of the test suite.

    za_monitoring_logging_archiving/
        'za' marks the first collection of tests that will be run here

        test_za_blah.py
            The first test module to run inside this 'za' folder

        test_za_bleh.py
            The second test module to run inside this 'za' folder

        ...

    zb_blabla/
        'zb' will mark the next collection of tests to run here

    ...

    zz_shutdown/
        'zz' marks the shutdown tests and should always be performed last

