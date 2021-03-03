import random
import nltk
import string


class MarkovModel:
    def __init__(self):
        self.model = None
    
    def learn (self, tokens, n=2):
        model = {}
        
        for i in range(0, len(tokens) - n):
            gram = tuple(tokens[i:i+n])
            token = tokens[i+n]
            
            if gram in model:
                model[gram].append(token)
                
            else:
                model[gram] = [token]
                
        finalGram = tuple(tokens[len(tokens) - n:])
        
        if finalGram in model:
            model[finalGram].append(None)
        
        else:
            model[finalGram] = [None]
            
        self.model = model
        return(model)
    
    
    def generate(self, n = 2, seed = None, maxTokens = 100):
        if seed == None:
            seed = random.choice(self.model.keys())

        output = list(seed)
        output[0] = output[0].capitalize()
        current = seed

        for i in range(n, maxTokens):
            # get next possible set of words from the seed word
            if current in self.model:
                possibleTransitions = self.model[current]
                choice = random.choice(possibleTransitions)
                if choice is None: break

                # check if choice is period and if so append to previous element
                if choice == '.':
                    output[-1] = output[-1] + choice
                else:
                    output.append(choice)
                current = tuple(output[-n:])
            else:
                # should return ending punctuation of some sort
                if current not in string.punctuation:
                    output.append('.')
        return(output)
    
    
    
    
    