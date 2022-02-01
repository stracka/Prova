import numpy as np
import struct

# decoding function 
def yfromfile(fName):

    with open(fName,'rb') as f:        

        f.seek(0); p=f.read(64).find(b'WAVEDESC')  # start of header
        
        f.seek(p+32); fmt=struct.unpack("H",f.read(2))[0]  # 1- or 2-byte word
        f.seek(p+34); en=struct.unpack("H",f.read(2))[0]   # endian-ness

        dp = 0  # size of header
        f.seek(p+36); dp+=struct.unpack("i",f.read(4))[0]
        f.seek(p+40); dp+=struct.unpack("i",f.read(4))[0]
        f.seek(p+48); dp+=struct.unpack("i",f.read(4))[0]
        f.seek(p+52); dp+=struct.unpack("i",f.read(4))[0]

        f.seek(p+60); wa=struct.unpack("i",f.read(4))[0]   # number of bytes

        f.seek(p+116); wac=struct.unpack("i",f.read(4))[0] # total number of samples 
        f.seek(p+144); sac=struct.unpack("i",f.read(4))[0] # number of records

        # for conversion to float: vgain * data - voffset
        f.seek(p+156); vg=struct.unpack("f",f.read(4))[0] # vgain 
        f.seek(p+160); vo=struct.unpack("f",f.read(4))[0] # voffset
        f.seek(p+176); hi=struct.unpack("f",f.read(4))[0] # sampling interval in seconds

        if(fmt):
            sfmt="int16"  # 2 bytes word
        else:
            sfmt="int8"   # 1 byte word

        # go to start of data array, then read
        f.seek(p+dp); y = np.fromfile(f, sfmt, wa)
       
    if(not en):
        y.byteswap(True) 
    
    dim1=int(wac/sac) # number of samples per record
    dim2=int(sac)
    
    return y.reshape(-1,dim1), hi*1e9, f.closed

# secondary pulse search 
def findap(arr, cc, indsh, offset):

    # arr    - input array (after baseline subtraction and cross-correlation filter)
    # cc     - response template w/ amplitude in [0,1] (after baseline subtraction and cross-correlation filter)
    # indsh  - position of the primary pulse 
    # offset - left offset of search range w.r.t. the primary pulse

    murfa = arr[indsh]                          # height of the primary pulse 
    arr = arr[indsh-offset:indsh-offset+cc.shape[0]]  # trim of the input array
      
    urfa = arr / murfa   
    subfa = urfa-cc

    t0=offset              # position of the primary pulse 
    t1=np.argmax(subfa)    # position of the secondary pulse

    # disentangle the two contributions
    f0=urfa[t0]
    f1=urfa[t1]

    dt=t1-t0
    cct0=np.argmax(cc)
    c0=cc[cct0]
    c1=cc[cct0+dt]

    a0 =  ( c0*f0 - c1*f1) / (c0**2 - c1**2) * murfa
    a1 =  (-c1*f0 + c0*f1) / (c0**2 - c1**2) * murfa

    # delta t and corrected amplitudes for secondary and primary pulses
    return dt,a1,a0
