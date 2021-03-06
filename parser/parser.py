import sys
import nltk
import re

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# TODO: this could be scaled to a higher level...
NONTERMINALS = """
S -> NP VP | VP NP | S Conj S
PP -> P NP
NP -> N | Det N | NP PP | Det AP N
VP -> V | V NP | V PP | Adv VP | VP Adv
AP -> Adj | Adj AP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # You may assume that sentence will be a string.
    # Okay, but just in case :)
    sentence = str(sentence)
    # Your function should return a list of words, where each word is a lower cased string.
    sentence = sentence.lower()
    # You should use nltk???s word_tokenize function to perform tokenization.
    sentence = nltk.word_tokenize(sentence)
    # Any word that does not contain at least one alphabetic character (e.g. . or 28)
    # should be excluded from the returned list.
    for word in sentence.copy():
        if not re.search("[a-z]", word):
            sentence.remove(word)

    return sentence


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # For this problem, a ???noun phrase chunk??? is defined as a noun phrase that does not contain other noun
    # phrases within it. Put more formally, a noun phrase chunk is a subtree of the original tree whose label
    # is NP and that does not itself #contain other noun phrases as subtrees.
    # You may assume that the input will be a nltk.tree object whose label is
    # S (that is to say, the input will be a tree representing a sentence).
    # Your function should return a list of nltk.tree objects, where each element has the label NP.
    # You will likely find the documentation for nltk.tree helpful for identifying how to manipulate a nltk.tree object.
    chunks = []

    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        # Stop at first level!
        if subtree[0].label() != "NP":
            chunks.append(subtree)

    return chunks


if __name__ == "__main__":
    main()
