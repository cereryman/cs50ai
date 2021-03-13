import nltk
import sys

import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Read data in from files
    words = {}
    # Your function should be platform-independent: that is to say, it should work regardless of operating system.
    # Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows.
    # Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.
    for file in os.listdir(directory):
        filename = os.path.join(directory, file)
        with open(filename) as f:
            if f is not None:
                # In the returned dictionary, there should be one key named for each .txt file in the directory.
                # The value associated with that key should be a string (the result of reading the corresponding file).
                # Each key should be just the filename, without including the directory name.
                # For example, if the directory is called corpus and contains files a.txt and b.txt, the keys should
                # be a.txt and b.txt and not corpus/a.txt and corpus/b.txt.
                words[file] = f.read().replace('\n', '')
            else:
                continue
    return words


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = str(document)
    # All words in the returned list should be lower cased.
    document = document.lower()
    # You should use nltk’s word_tokenize function to perform tokenization.
    document = nltk.word_tokenize(document)
    # Filter out punctuation and stopwords (common words that are unlikely to be useful for querying).
    # Punctuation is defined as any character in string.punctuation (after you import string).
    punctuation = string.punctuation
    # Stopwords are defined as any word in nltk.corpus.stopwords.words("english").
    stopwords = set(nltk.corpus.stopwords.words("english"))

    tokenized_words = []
    for word in document:
        if word not in punctuation and word not in stopwords:
            tokenized_words.append(word)

    return tokenized_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Assume that documents will be a dictionary mapping names of documents to a list of words in that document.
    idfs = {}
    unique_words = []
    total_documents = len(documents)

    for key, value in documents.items():
        for word in value:
            unique_words.append(word)
    unique_words = set(unique_words)

    # The returned dictionary should map every word that appears in at least one of the
    # documents to its inverse document frequency value.
    for word in unique_words:
        no_documents = 0
        for key, value in documents.items():
            if word in value:
                no_documents = no_documents + 1

        # Recall that the inverse document frequency of a word is defined by taking the natural logarithm of the
        # number of documents divided by the number of documents in which the word appears.
        idfs[word] = math.log(total_documents/no_documents)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    files_total_tfidf = {}

    # Files should be ranked according to the sum of tf-idf values for any word in the query that also
    # appears in the file. Words in the query that do not appear in the file should not contribute to the file’s score.
    for key, value in files.items():
        tfidf = 0
        for word in query:
            count = value.count(word)
            # Recall that tf-idf for a term is computed by multiplying the number of times the term appears in the
            # document by the IDF value for that term.
            tfidf = tfidf + count * idfs.get(word, 0)
        files_total_tfidf[key] = tfidf

    # The returned list of filenames should be of length n and should be ordered with the best match first.
    # You may assume that n will not be greater than the total number of files.
    top_files_output = sorted(files_total_tfidf.items(), key=lambda x: (x[1]), reverse=True)[:n]
    top_files_list_output = []
    for sentence in top_files_output:
        top_files_list_output.append(sentence[0])

    return top_files_list_output


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_total_tfidf = {}

    for key, value in sentences.items():
        query_update = query.intersection(value)
        idf = 0
        for word in query_update:
            # Sentences should be ranked according to “matching word measure”: namely, the sum of IDF values for
            # any word in the query that also appears in the sentence. Note that term frequency should not be taken
            # into account here, only inverse document frequency.
            idf = idf + idfs.get(word, 0)

        # If two sentences have the same value according to the matching word measure, then sentences with a
        # higher “query term density” should be preferred. Query term density is defined as the proportion of words
        # in the sentence that are also words in the query. For example, if a sentence has 10 words, 3 of which are in
        # the query, then the sentence’s query term density is 0.3.
        no_words = len(value)
        count_of_query_term = 0
        for word in query_update:
            count_of_query_term = count_of_query_term + value.count(word)
        query_term_density = count_of_query_term / no_words

        sentence_total_tfidf[key] = {'idf': idf, 'qtd': query_term_density}

    # The returned list of sentences should be of length n and should be ordered with the best match first.
    # You may assume that n will not be greater than the total number of sentences.
    top_sentences_out = sorted(sentence_total_tfidf.items(), key=lambda x: (x[1]['idf'], x[1]['qtd']), reverse=True)[:n]
    top_sentences_list_output = []
    for sentence in top_sentences_out:
        top_sentences_list_output.append(sentence[0])

    return top_sentences_list_output


if __name__ == "__main__":
    main()
