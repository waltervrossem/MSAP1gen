# -*- coding: iso-8859-1 -*-

#============================================================
#
#           This file is part of FEval, a module for the
#           evaluation of Finite Element results
#
# Licencse: FEval is provided under the GNU General Public License (GPL)
#
# Authors:  Martin Lthi, tinu@tnoo.net
#
# Homepage: http://feval.sourceforge.net
#
# History:  2001.09.21 (ml):  Code cleaned up for intial release
#           2002.08.15 (ml):  Speed improvement of 20% when byteswapping is
#                             hardcoded (but less elegant)
#
#============================================================

# Fortran binary IO (at present only input of array)
# The handling of zipped files is based on Konrad Hinsen's TextFile

from __future__ import print_function
from builtins import object
import os, string, sys
import struct
import numpy as N

# this is taken from the Numeric.py module, where it is defined globally
# LittleEndian = Numeric.fromstring("\001"+"\000"*7, 'i')[0] == 1

def myFromstring(endian):
    """Handle the problems caused by endian types.
    o Big endian:    Sun Sparc, HP etc.
    o Little endian: Intel machines (PC)
    Give here the endian type of the file, the endian type of the machine
    is automatically determined by the constant LittleEndian above.
    The conventions are as those given in 'struct' module:
    '@','=' : File is native (was produced on the same machine)
    '<'     : File is little endian
    '>','!' : File is big endian (network)
    """
    littleendian = False
    try:
        if N.LittleEndian:
            littleendian = True
    except:
        if N.little_endian:
            littleendian = True
            
    if littleendian:
        if endian in ['>','!']:
            return 'swap', lambda s, t: N.fromstring(s,t).byteswap()
        else:
            return 'noswap', N.fromstring
    elif endian in ['<']:
        return 'swap', lambda s, t: N.fromstring(s,t).byteswap()
    else:
        return 'noswap', N.fromstring


class FortranBinaryFile(object):
    """Fortran binary file support
    It is assumed that the file consists of Fortran records only
    Constructor: FortranBinaryFile(filename, mode)
    """

    def __init__(self, filename, mode='r', endian='@', verbose=0):
        self.filename = filename = os.path.expanduser(filename)
        self.endian = endian
        self.swap, self.fromstring = myFromstring(endian)
        self.verbose = verbose
        if self.verbose:
            print('the byte-swapping is', self.swap)
        # this accelerates the file reading by about 20%
        '''
        if self.swap=='swap':
            FortranBinaryFile.__dict__['readRecord'] = FortranBinaryFile.__dict__['readRecordByteswapped']
        else:
            FortranBinaryFile.__dict__['readRecord'] = FortranBinaryFile.__dict__['readRecordNative']
        self.readRecord = self.readRecordByteswapped
        '''
        if (mode == 'r' or mode == 'rb' ):
            if not os.path.exists(filename):
                raise IOError(2, 'No such file or directory: '+ filename)
            if filename[-2:] == '.Z':
                self.file = os.popen("uncompress -c " + filename, 'rb')
            elif filename[-3:] == '.gz':
                self.file = os.popen("gunzip -c " + filename, 'rb')
            else:
                try:
                    self.file = open(filename, mode)
                except IOError as details:
                    if type(details) == type(()):
                        details = details + (filename,)
                    raise IOError(details)

            self.filesize = os.path.getsize(filename)

        elif ( mode == 'w' or mode == 'w'):
            if filename[-2:] == '.Z':
                self.file = os.popen("compress > " + filename, mode)
            elif filename[-3:] == '.gz':
                self.file = os.popen("gzip > " + filename, mode)
            else:
                try:
                    self.file = open(filename, 'wb')
                except IOError as details:
                    if type(details) == type(()):
                        details = details + (filename,)
                    raise IOError(details)
        else:
            raise IOError(0, 'Illegal mode: ' + repr(mode))

    def __del__(self):
        self.close()

    def close(self):
        self.file.close()

    def flush(self):
        self.file.flush()

    def readRecordNative(self, dtype=None):
        a = self.file.read(4)   # record size in bytes
        recordsize = N.fromstring(a,'i')
        record = self.file.read(recordsize[0])
        self.file.read(4)   # record size in bytes

        if dtype in ('f', 'i', 'I', 'b', 'B', 'h', 'H',  'l', 'L', 'd'):
            return N.fromstring(record,dtype)
        elif dtype in ('c', 'x'):
            return struct.unpack(self.endian+'1'+dtype, record)
        else:
            return (None, record)

    def readRecordByteswapped(self, dtype=None):
        a = self.file.read(4)   # record size in bytes
        recordsize = N.fromstring(a,'i').byteswap()
        record = self.file.read(recordsize[0])
        self.file.read(4)   # record size in bytes

        if dtype in ('f', 'i', 'I', 'b', 'B', 'h', 'H',  'l', 'L', 'd'):
            return N.fromstring(record,dtype).byteswap()
        elif dtype in ('c', 'x'):
            return struct.unpack(self.endian+'1'+dtype, record)
        else:
            return (None, record)

    def readBytes(self, recordsize, dtype, offset = None):
        if offset:
            self.file.seek(offset)
            record = self.file.read(recordsize*struct.calcsize(dtype))
        if dtype in ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'f', 'd'):
            return self.fromstring(record, dtype)
        elif dtype in ('c', 'x'):
            return struct.unpack(self.endian+'1'+ttype, record)
        else:
            return (None, record)

### Test

if __name__ == '__main__':
    pass


