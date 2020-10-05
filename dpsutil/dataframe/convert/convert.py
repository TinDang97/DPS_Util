from binascii import unhexlify, hexlify

from dpsutil.common import check_type_instance


def cvt_str2byte(string):
    check_type_instance(string, str)
    return ''.join(map(lambda x: f"{ord(x):08b}", string)).encode()


def cvt_8bit(binary):
    check_type_instance(binary, bytes)
    return ''.join(map(lambda x: f"{x:08b}", binary)).encode()


def cvt_str2hex(string):
    check_type_instance(string, str)
    return unhexlify(''.join(map(lambda x: f"{ord(x):x}", string)))


def cvt_dec2hex(number):
    output = f"{number:02x}"
    if len(output) % 2:
        output = f"0{output}"
    return unhexlify(output)


def cvt_hex2dec(binary):
    return int(hexlify(binary), 16)


def cvt_byte2str(binary):
    check_type_instance(binary, bytes)
    if len(binary) % 8:
        raise ValueError("Bytes was broken!")
    return ''.join(chr(int(binary[i:i + 8], 2)) for i in range(0, len(binary), 8))


def cvt_hex2str(binary):
    check_type_instance(binary, bytes)
    return ''.join(chr(i) for i in binary)


def cvt_bin2hex(binary):
    check_type_instance(binary, bytes)
    return unhexlify(''.join(f"{int(binary[i:i + 8], 2):x}" for i in range(0, len(binary), 8)))
