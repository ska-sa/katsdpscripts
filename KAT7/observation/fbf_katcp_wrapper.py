"""
Client for communicating with the beamformer receiver on kat-dc2.karoo

Author: R van rooyen
Date: 2014-01-03
Modified:

"""

from katcp import *

import logging
log = logging.getLogger("katcp")


class FBFClient(BlockingClient):
# class FBFClient(CallbackClient):
    """Client for communicating Beamformer receiver

       Notes:
         - All commands are blocking.
         - If there is no response to an issued command, an exception is thrown
           with appropriate message after a timeout waiting for the response.
         - If the TCP connection dies, an exception is thrown with an
           appropriate message.
       """
    def __init__(self, host, port=7147, tb_limit=20, timeout=10.0, logger=log):
        """Create a basic DeviceClient.

           @param self  This object.
           @param host  String: host to connect to.
           @param port  Integer: port to connect to.
           @param tb_limit  Integer: maximum number of stack frames to
                            send in error traceback.
           @param timeout  Float: seconds to wait before timing out on
                           client operations.
           @param logger  Object: Logger to log to.
           """
        super(FBFClient, self).__init__(host, port, tb_limit=tb_limit,timeout=timeout, logger=logger)
        self.host=host
        self._timeout = timeout
        #self.start(daemon=True)
        self.start()
        print "IT's CHRIS!"

    def inform_log(self,message):
        "If we get a log inform, log it."
        DeviceLogger.log_to_python(self._logger, message)

    def _request(self, name, *args):
        """Make a blocking request and check the result.

           Raise an error if the reply indicates a request failure.

           @param self  This object.
           @param name  String: name of the request message to send.
           @param args  List of strings: request arguments.
           @return  Tuple: containing the reply and a list of inform messages.
           """
        request = Message.request(name, *args)
        timeout=3600.0 # Nr of seconds to wait before timing out on client operation
        reply, informs = self.blocking_request(request, timeout = timeout, keepalive=True)
#         reply, informs = self.blocking_request(request,keepalive=True)

        if reply.arguments[0] != Message.OK:
            self._logger.error("Request %s failed.\n  Request: %s\n  Reply: %s."
                    % (request.name, request, reply))

            raise RuntimeError("Request %s failed.\n  Request: %s\n  Reply: %s."
                    % (request.name, request, reply))
        return reply, informs

    def rx_init(self, prefix, halfband=False, transpose=False):
        """Initialise beamformer receiver and set up environment variables for capture

           @param self     This object.
           @param prefix    String: Data output directory
           @param halfband  Boolean: [Optional]Set to record only inner 50% of the band
           @param transpose Boolean: [Optional]Set to transpose time frequency blocks
           @return      Boolean: Reply message indicating success.
           """
        try: reply, informs = self._request("rx-init", prefix, int(not halfband), int(transpose))
        except: raise
        if reply.arguments[0]=='ok': return True
        else: return False

    def rx_close(self):
        """Closing beamformer receiver

           @param self  This object.
           @return      String: Directory name where captured data is housed
           """
        reply, informs = self._request("rx-close")
        if reply.arguments[0]=='ok':
            return reply.arguments[-1]
        else: raise RuntimeError('Cannot move output directory, verify data in PREFIX directory')

    def rx_meta_init(self, port):
        """Start the receiver to capture beamformer meta data

           @param self       This object.
           @param port       String: Selected port to output metadata
           @return      String: Metadata output port used
           """
        reply, informs = self._request("rx-meta-init", int(port))
        if reply.arguments[0]=='ok': return str(informs[0]).split()[-1].replace('\_'," ")

    def rx_meta(self, meta_dict=None):
        """Capture beamformer meta data after transmit has started

           @param self       This object.
           @param meta_dict  Dictionary: Observation specific metadata # temp fix until augmentation of beamformer data
           @return      Boolean: Reply message indicating success.
           """
        meta_str = ''
        if meta_dict is not None:
            for key in meta_dict.keys():
                meta_str += ('%s: %s;' % (key, str(meta_dict[key])))
        reply, informs = self._request("rx-meta", meta_str)
        if reply.arguments[0]=='ok': return True
        else: raise RuntimeError('Failure to capture meta data')

    def rx_beam(self, pol='h', port='7150'):
        """Capture beamformer data

           @param self            This object.
           @param pol             String: Polarization associated with beam
           @param port            String: Selected port to output metadata
           @return      String: Data output port used
           """
        reply, informs = self._request("rx-beam", int(port), pol)
        if reply.arguments[0]=='ok': return str(informs[0]).split()[-1].replace('\_'," ")

    def rx_stop(self):
        """Safely stops all SPEAD receivers and tear down client connections

           @param self            This object.
           @return      String: Data output port used
           """
        reply, informs = self._request("rx-stop")
        if reply.arguments[0]=='ok': return str(informs[0]).split()[-1].replace('\_'," ")

# -fin-

