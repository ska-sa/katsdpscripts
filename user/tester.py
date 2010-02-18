# Configure DBE and other test functions (temporary)

def configure_dbe (packet_count, dump_rate, destination, name):
    """Function to configure DBE and start capture.

    Parameters
    ----------
    packet_count : integer
        Test for integer parameter.
    dump_rate : float
        Test for float parameter.
    destination : string
        Test for string parameter.
    name : string
        Test for name parameter.

    Returns
    -------
    : Boolean
        Indication of success or failure
    """

    try:
        kat.dbe.req.dbe_packet_count(packet_count)
        kat.dbe.req.dbe_dump_rate(dump_rate)
        kat.dbe.req.capture_start(name)
        return True
    except Exception, err:
        print "configure_dbe failed -->",err
        return False


def noise_coupler_on (period, duty_cycle):
    """Switch noise coupler on with specified period and duty_cycle on ant1 and ant2.

    Parameters
    ----------
    period : integer
        Period for modulation.
    duty_cycle : float
        Duty cycle of modulation.

    Returns
    -------
    : Boolean
        Indication of success or failure
    """

    try:
        kat.ped1.req.rfe3_rfe15_noise_source_on("coupler", 1, "now", 1, period, duty_cycle)
        kat.ped2.req.rfe3_rfe15_noise_source_on("coupler", 1, "now", 1, period, duty_cycle)
        return True
    except Exception, err:
        print "noise_coupler_on failed -->",err
        return False
