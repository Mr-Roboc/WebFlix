import string

def words():

   try:
       with open("data/stopwords.txt","r") as f:
              words=f.read()

              word_list =  words.splitlines()

#              print("Before Preprocessing:", word_list) 

       # Preprocessing

              preprocessed_words = list(map(lambda text: text.translate(str.maketrans("","",string.punctuation)), word_list))

#              print("After preprocessing:")

              return preprocessed_words

   except FileNotFoundError:
          print('FIle not found')
