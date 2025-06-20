# 待编码字符串
input_string = 'ababcababac'

# 初始化字典，包含 ASCII 编码表中的所有字符
dictionary = {chr(i): i for i in range(256)}
next_code = 256
p = ""
encoded = []

# 编码过程
for c in input_string:
    pc = p + c
    if pc in dictionary:
        p = pc
    else:
        encoded.append(dictionary[p])
        dictionary[pc] = next_code
        next_code += 1
        p = c

if p:
    encoded.append(dictionary[p])

print("编码结果:", encoded)
# 编码结果: [97, 98, 256, 99, 256, 260, 99]

# 解码结果: ababcababac
# 初始化解码字典，包含 ASCII 编码表中的所有字符
decoding_dictionary = {i: chr(i) for i in range(256)}
next_decode_code = 256
p = chr(encoded.pop(0))
decoded = [p]

# 解码过程
for code in encoded:
    if code in decoding_dictionary:
        entry = decoding_dictionary[code]
    elif code == next_decode_code:
        entry = p + p[0]
    else:
        raise ValueError("Bad compressed code")
    decoded.append(entry)
    decoding_dictionary[next_decode_code] = p + entry[0]
    next_decode_code += 1
    p = entry

decoded_string = ''.join(decoded)
print("解码结果:", decoded_string)
