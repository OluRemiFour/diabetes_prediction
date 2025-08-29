# import arc
# four_sqr = 12
#
# print(arc.cube(four_sqr))
# print(arc.square(2))

import random
import string

chars = ' ' + string.ascii_letters + string.punctuation + string.digits
chars = list(chars)
key = chars.copy()

print(chars)

random.shuffle(key)

# Encrypt
plain_text = (input('Enter a message to encrypt: '))
cypher_text = ''

for letter in plain_text:
    index = chars.index(letter)
    cypher_text += key[index]

print(f'original message: {plain_text}')
print(f'encrypted message: {cypher_text}')


# Decrypt
cypher_text = (input('Enter a message to decrypt: '))
plain_text = ''

for letter in cypher_text:
    index = key.index(letter)
    plain_text += chars[index]


print(f'encrypted message: {cypher_text}')
print(f'original message: {plain_text}')
