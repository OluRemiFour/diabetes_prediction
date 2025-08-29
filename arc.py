# item = input('What item would you like to buy?: ')
# price = float(input('How much is it?: '))
# quantity = int(input('How many would you like?: '))
# total_price = price * quantity
#
# print(total_price)

# firstName = 'bro'
#
# print(f"hello {firstName}")

# import math

# print(math.pi)
# print(math.e)

# credit_card = '1234-5678-9012-3456'
# print(credit_card[0:4])

# questions = (
#     'How many elements are in the perodic table?.',
#     'Which animal lays the largest eggs?',
#     'How many bones are in human body'
# )
#
# options = (
#     ('A.116', 'B.124', 'C.115', 'D.405'),
#     ('A. whale', 'B. Cheeta', 'C. Dog'),
#     ('A. 206', 'D. 102', 'C. 23'),
# )
#
# answers = ("C", "D", "A",)
# guesses = []
# score = 0
# question_number = 0
#
# for question in questions:
#     print('---------------')
#     print(question)
#     for option in options[question_number]:
#         print(option)
#
#     guess = input('Enter your guess: ')
#     guesses.append(guess)
#     if guess == answers[question_number]:
#         score += 1
#         print('Correct!')
#     else:
#         print('Incorrect!')
#         print(f'Correct answer: {answers[question_number]}')
#     question_number += 1
#
# print(f'your score is {score}')
#
# print('Answers: ---------')
# for answer in answers:
#     print(answer)
#
# print('Guesses: ---------')
# for guess in guesses:
#     print(guess)
#

# Dictionaries
# capitals = {'usa' : 'wDC', 'india': 'new delhi', "nigeria": "Abuja" }
#
# for key, value in capitals.items():
#     # print(capitals[key])
#     print(f'{value} {key}')

# DICE ------------------------------

# import random
# # ● ┌ ─ ┐ │ └ ┘
# "┌──────────┐"
# "│"        "│"
# "│"        "│"
# "│"        "│"
# "└──────────┘"
#
# dice_art = {
#     1: ("┌──────────┐",
#         "│          │",
#         "│    ●     │",
#         "│          │",
#         "└──────────┘"),
#     2: ("┌──────────┐",
#         "│  ●       │",
#         "│          │",
#         "│     ●    │",
#         "└──────────┘"),
#     3: ("┌──────────┐",
#         "│  ●       │",
#         "│     ●    │",
#         "│       ●  │",
#         "└──────────┘"),
#     4: ("┌──────────┐",
#         "│  ●    ●  │",
#         "│          │",
#         "│  ●    ●  │",
#         "└──────────┘"),
#     5: ("┌──────────┐",
#         "│  ●    ●  │",
#         "│     ●    │",
#         "│  ●    ●  │",
#         "└──────────┘"),
#     6: ("┌──────────┐",
#         "│  ●    ●  │",
#         "│  ●    ●  │",
#         "│  ●    ●  │",
#         "└──────────┘"),
# }
# dice = []
# total = 0
# num_of_dice = int(input('How many dice?: '))
#
# for die in range(num_of_dice):
#     dice.append(random.randint(1, 6))
#
# for line in range(5):
#     for die in dice:
#         print(dice_art.get(die)[line], end="")
#     print()
#
# for die in dice:
#     total+=die
#     # print(total)

# Function --------------------
def add(*args):
    for arg in args:
        print(arg, end=" ")


def print_address(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# print_address(street='123, Fake St.', city='Fake City', state='Fake State')

def shipping_label(*args ,**kwargs):
    for arg in args:
        print(arg, end=" ")

    for kwarg in kwargs:
        print(f"{kwarg}: {kwargs[kwarg]}")

# shipping_label("Dr. Abiola Balogun II", street="Pipeline", city="Ilorin", country="Brazil")

# Membership operators ----------------
# word = 'MIMI'
# letter = input('Input your letter: ')
#
# if letter in word.lower():
#     print(f'{letter} was found')
# else:
#     print(f'{letter} was NOT found')
#
#

doubles = [x * 2 for x in range(2, 11)]
# print(doubles)

# numbers = [2,3,4,5,6]
# for num in numbers:
#     print(num + 1, end=' ')

# add = [num + 1 for num in numbers]
# print(add)

# print(help('modules'))

pi = 3.14
def cube(x):
    return x ** 3

def square(x):
    return x ** 2

