from phe import paillier
import json


public_key, private_key = paillier.generate_paillier_keypair()

def encrypt(input):

    encrypted_number_list = [public_key.encrypt(x) for x in input]
    enc_with_one_pub_key = {}
    enc_with_one_pub_key['public_key'] = {'g': public_key.g, 'n': public_key.n}
    enc_with_one_pub_key['values'] = [(str(x.ciphertext()), x.exponent) for x in encrypted_number_list]
    json_content = json.dumps(enc_with_one_pub_key)

    return json_content

def serialize(data):

    json_content = json.dumps(data)
    return json_content

