import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def multiplicative_inverse(e, phi):
    d = 0
    x1 = 0
    x2 = 1
    y1 = 1
    temp_phi = phi

    while e > 0:
        temp1 = temp_phi//e
        temp2 = temp_phi - temp1 * e
        temp_phi = e
        e = temp2

        x = x2 - temp1 * x1
        y = d - temp1 * y1

        x2 = x1
        x1 = x
        d = y1
        y1 = y

    if temp_phi == 1:
        return d + phi

def is_prime(num):
    if num == 2:
        return True
    if num < 2 or num % 2 == 0:
        return False
    for n in range(3, int(num**0.5)+2, 2):
        if num % n == 0:
            return False
    return True

def generate_key_pair(p, q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError('Both numbers must be prime.')
    elif p == q:
        raise ValueError('p and q cannot be equal')
    n = p * q

    # Phi is the totient of n
    phi = (p-1) * (q-1)

    # Choose an integer e such that e and phi(n) are coprime
    e = random.randrange(1, phi)

    # Use Euclid's Algorithm to verify that e and phi(n) are coprime
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)

    # Use Extended Euclid's Algorithm to generate the private key
    d = multiplicative_inverse(e, phi)

    # Return public and private key_pair
    # Public key is (e, n) and private key is (d, n)
    return ((e, n), (d, n))

def encrypt(pk, plaintext):
    # Unpack the key into it's components
    key, n = pk
    # Convert each letter in the plaintext to numbers based on the character using a^b mod m
    cipher = [pow(ord(char), key, n) for char in plaintext]
    # Return the array of bytes
    return cipher
def Convert(string_series):
    floats_list = list(string_series.split(" "))
    return floats_list

data = pd.read_csv("C:/Users/Admin/projects/dps/models/stress.csv")
p=19
q=23
public,private = generate_key_pair(p,q)
encrypteddata= encrypt(public,str(data))
X = data.iloc[:,:-1].values
y = data['sl']
#X = encrypt(X)
#y = encrypt(y)

encrypted_X = encrypt(public,str(X))
encrypted_y = encrypt(public,str(y))

#print('x:',encrypted_X)
#print('y',encrypted_y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=27)
KNN_model = KNeighborsClassifier(n_neighbors=7)
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)
print(X_test)

#print(KNN_prediction)
print(accuracy_score(KNN_prediction, y_test))

filename = 'finalized-model.sav'
joblib.dump(KNN_model,filename)