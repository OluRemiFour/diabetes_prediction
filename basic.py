# print("    /\")
# print("   /  \")
# print("  /    \")
# print(" /      \")
# print("/________\")

character_name = 'John'
character_age = '35'

phrase = "Giraffe\nAcademy"
print(phrase.upper())
print(len(phrase))

# Getting input from users

# name = input('Enter your name: ')
# age = input('Enter your age: ')
# print("Hello " + name + ' I am ' + age, 'years old')

plura = input("input plura:")
color = input("input color name:")
celebrity = input("input celebrity name:")

print("Roses are", color)
print(plura, "are blue")
print("I love", celebrity)

# Lists --------------------- 
# friends = ['muke', 'josh', 'toni']
# lucky_numbers = [3,4,5,6,7,8]

# this will add luck_numbers to the list of friends 
friends.extens(lucky_numbers)

# this will append 'Creed' to the list of friends
# friends.append("Creed")

# this take 2 parameter, index position and value 
# friends.insert(1, 'Kelly')

# friends.remove('toni')
# friends.clear()
# friends2 = friends.copy()

# sorts order of the list
# luck_numbers.sort()

# reverse the order of the list
# luck_numbers.reverse()

def cube(num): 
    return num * num 

print(cube(3))

# Dictionaries
monthConversions = {
    'jan': 'january',
    'feb': 'february',
    'mar': 'march',
    1: 'orange',
    14: 'banana'
}

print(monthConversions.get('man', 'Not a valid dictionary key'))


# while loop
i = 1
while i <= 10:
    print(i);
    i += 1;

print('done with looping')

secrect_word = 'giraffe'
guess = ''
guess_count = 0,
guess_limit = 3,
out_of_guess = False;

while secrect_word != guess and not(out_of_guess):
    if guess_count < guess_limit:
        guess = input("Enter guess: ")    
        guess_count += 1
    else: 
        out_of_guess == True
    
if out_of_guess:
    print('Out of guess, You damn LOOSE')
else:
    print('You win!')

# For loop
friends = ['jim', 'ike', 'bayo', 'mustie']
for friend in friends:
    print(friend)

# 2D Listing (accessing data in 2d lists)
# number_grid = [
#     [1, 2, 3, 4],
#     [11, 22, 33, 44],
#     [01, 02, 03, 04]
#     [0]
# ]

# print(number_grid[2][0])

try: 
    number = int(input('Enter a number: '))
    print(number)
except: 
    print('Invalid input')

# Reading file from file system
employee_file = open('employees.txt', 'r')
print(employee_file.read())
employee_file.close()

# Appending file to file system (a): append
employee_file = open('employees.txt', 'a')
employee_file.write('\n Musa - Software Engineer')
employee_file.close()

# Writing file to file system (w): write
employee_file = open('employees.txt', 'w')
employee_file.write('<p> Musa - Software Engineer </p>')
employee_file.close()