import math

def log2(x):
    return math.log(x)/math.log(2)

def tobin(i):
    """
    Maps a ray index "i" into a bin index.
    """
    #return int(log(3*y+1)/log(2))>>1
    return int((3*i+1).bit_length()-1)>>1

def findMSB(x):
    if x == 0 or x == -1:
        return -1
    return x.bit_length()-1

def tobin2(i):
    """
    Maps a ray index "i" into a bin index.
    """
    #return int(log(3*y+1)/log(2))>>1
    return int(findMSB(3*i+1))>>1

def binto(b):
    """
    Maps a bin index into a starting ray index. Inverse of "tobin(i)."
    """
    return (4**b - 1) // 3

def z2x_1(x):
    x = x & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF
    return x

def z2xy(z):
    "Maps 32-bit Z-order index into 16-bit (x, y)"
    return (z2x_1(z), z2x_1(z>>1))

def xy2z(x, y):
    """
    Interleave lower 16 bits of x and y, so the bits of x
    are in the even positions and bits from y in the odd;
    z gets the resulting 32-bit Morton Number.
    x and y must initially be less than 65536.

    Source: http://graphics.stanford.edu/~seander/bithacks.html
    """

    B = [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF]
    S = [1, 2, 4, 8]

    x = (x | (x << S[3])) & B[3]
    x = (x | (x << S[2])) & B[2]
    x = (x | (x << S[1])) & B[1]
    x = (x | (x << S[0])) & B[0]

    y = (y | (y << S[3])) & B[3]
    y = (y | (y << S[2])) & B[2]
    y = (y | (y << S[1])) & B[1]
    y = (y | (y << S[0])) & B[0]

    z = x | (y << 1);
    return x | (y << 1)

def i2ray(i):
    b = tobin(i)
    start = binto(b)
    z = i - start
    x, y = z2xy(z)
    dim = 2**b
    parent_size = (2**(b-1))**2
    parent = int(start - parent_size) + (z//4)

    sp = 1/dim
    u = (1/2) * (1/dim) + (1 - 1/dim) * x
    v = (1/2) * (1/dim) + (1 - 1/dim) * y
    #print(f"parent_size: {parent_size}")
    #print(f"[{i}] b: {b}, dim: {dim}, parent: {parent}, z: {z}, size: {dim}x{dim}", end='')
    #print(f" ({x}, {y}) -> ({u:.3f}, {v:.3f})")
    print(f"[{i}] ({x}, {y}) -> ({u:.5f}, {v:.5f}), parent index: {parent}")

def dim2nodecount(dim):
    """
    How many nodes must the whole hierarchy have, when leaf layer
    has "dim" nodes.
    """
    return binto(int(math.ceil(log2(dim))) + 1)

i2ray(0)
i2ray(3)
i2ray(5)
i2ray(6)

