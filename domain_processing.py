#Input layer and vocab one hot encoded inputs
#Embedding layer and its resultant transformation into vocab size<- another parameter
#Noisy inputs, etc 
import csv 
import string
import pickle
import pandas as pd
import nltk 
import spacy
import string

nlp = spacy.load('en') #load spacy model

    
def obtain_tokens(sentence):
    return sentence.split()

start_tag = "<START>"
end_tag = "<END>"
tag_to_id = {start_tag:0,end_tag:-1, "BA":1, "IA":2, "BO":3, "IO":4, "OT":5}
id_to_tag = {id_: tag for tag,id_ in tag_to_id.items()}

'''The below functions that use Spacy can be made faster using Spacy specific syntax- example using doc, checking for verbs, etc. However, here it is converted to Python string format to make the primary function compatible with other NLP packages'''
def get_tokenized_list_for_delim(phrases, tokenizer_func = obtain_tokens, delim =";"):
    '''
    Input: A sequence of phrases delimited as provided by delimiter: Ex A B; C; D E F
    Output: A list of tokenized phrases: [[A, B], [C], [D, E, F]]
    '''
    #split_phrases = phrases.split(delim)
    
    split_phrases = phrases.split(delim)
    tokenized_split_phrases = map(lambda x: tokenizer_func(x), split_phrases)
    return tokenized_split_phrases


punctuation_dict = {'.':' <PERIOD> ',
                    ',':' <COMMA> ', 
                    '"':' <QUOTATION_MARK> ', 
                    ';':' <SEMICOLON> ',
                    '!': ' <EXCLAMATION_MARK> ',
                    '?': ' <QUESTION_MARK> ', 
                    ':': ' <COLON> '
                   ,'-': ' <HYPHEN> ',
                    '(': ' <LEFT PARENTHESIS> ',
                    ')': ' <RIGHT PARENTHESIS> '
                   }


def process_text_pos_and_punctuation(text, replace_punctuation):
    text = nlp(unicode(text))
    tagged_text = []
    tokenized_text = []
    if(replace_punctuation):
        for token in text:
            temp_token = token
            if(str(token.pos_) == 'PUNCT'):
                if(str(token) in punctuation_dict.keys()):
                    token = punctuation_dict[str(token)]
            tokenized_text.append(str(token))
            tagged_text.append((str(token), str(temp_token.pos_), str(temp_token.tag_), str(temp_token.dep_)))
    
    else:
        tagged_text = [(str(token), str(token.pos_), str(token.tag_), str(token.dep_)) for token in text]
        tokenized_text = map(lambda x:x[0], tagged_text)
    return tokenized_text, tagged_text

def store_dataset_info(df, save_to_file = False, save_path ='Final_data/laptop', include_sense = False,  replace_punctuation= True, to_lower = True, get_tags = True, remove_punctuation= False, get_dependency_structure = False, tokenizer_func= obtain_tokens):
    training_data = []
    training_data_with_other_stuff = []
    out_row = []
    token_to_freq = {} 
    #token_sense_to_id = {}
    
    token_to_id = {}
    loop_i = 0 
    for sentence, aspect_words, opinion_words in zip(df.Sentence, df.Aspects, df.Opinions):
        
        
        if(remove_punctuation):
            sentence = sentence.translate(None, string.punctuation)
        
        
        if(to_lower):
            sentence = sentence.lower()
            
        
        tokenized_sentence, tagged_text = process_text_pos_and_punctuation(sentence, replace_punctuation)

        if(get_dependency_structure):
            #Obtain dependency tree
            dependency_tree = get_dependency_tree(sentence)
        #if(get_parser_tree):
            #Obtain parser tree
         #   None
        
        
        aspect_list = get_tokenized_list_for_delim(aspect_words, tokenizer_func, ';')
        opinion_list = get_tokenized_list_for_delim(opinion_words, tokenizer_func, ';')
        #print(aspect_list)
        seq_absa_tagged = [] 
        aspect_ptr = 0
        opinion_ptr = 0
        skip = 0
        
        #labels = []
        for i_loop, token in enumerate(tokenized_sentence):
            temp_token = token 
            
            #1) If sense is included then we change token to token|POS
            if(include_sense):
                '''THIS is specific to spacy format'''
            
                #Only limit to nouns, verbs and adjectives
                
                if(tagged_text[i_loop][1] in ['NOUN','VERB','ADJ']):
                    token+= '|'+ tagged_text[i_loop][1]
                    tokenized_sentence[i_loop] = token
                    #print(token)
                    
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
                token_to_freq[token] = 0 
            else:
                token_to_freq[token] += 1
                
                
                
            if(skip>0): #if we encounter incomplete aspect/opinion matches previously
                skip -= 1
             
            else:    
                label = [tag_to_id["OT"]] #assume it is OT, store as list in case of multiple aspect/opinion terms
        
        
                token = temp_token
                
                if(aspect_ptr < len(aspect_list) and token == aspect_list[aspect_ptr][0]): #Incomplete match: battery--> battery life
                    label = [tag_to_id["BA"]]
                    skip = len(aspect_list[aspect_ptr]) - 1 #words to skip ahead since they have been already covered
                    if(skip>0):
                        label += [tag_to_id["IA"] for i in aspect_list[aspect_ptr][1:]] 
                    aspect_ptr += 1 
        
                elif(opinion_ptr< len(opinion_list) and token == opinion_list[opinion_ptr][0]): 
                    label = [tag_to_id["BO"]]
                    skip = len(opinion_list[opinion_ptr]) - 1
                    if(skip>0):
                        label += [tag_to_id["IO"] for i in opinion_list[opinion_ptr][1:]]
                    opinion_ptr += 1 
        
            
                seq_absa_tagged+=label
                
        if(get_dependency_structure):
            training_data_with_other_stuff.append((tokenized_sentence, seq_absa_tagged, tagged_text, dependency_tree))
            out_row = ["Sentence", "Sequence", "Tags","Dependency Tree"]
        
        elif(get_tags):
            training_data_with_other_stuff.append((tokenized_sentence, seq_absa_tagged, tagged_text))
            out_row = ["Sentence", "Sequence", "Tags"]
        
        elif(get_dependency_structure):
            training_data_with_other_stuff.append((tokenized_sentence, seq_absa_tagged, dependency_tree))
            out_row = ["Sentence", "Sequence", "Dependency Tree"]
        else:
            training_data_with_other_stuff.append((tokenized_sentence, seq_absa_tagged))
            out_row = ["Sentence", "Sequence"]
            
        training_data.append((tokenized_sentence, seq_absa_tagged))
        #Write data to csv and save vocab as pickle
        
        if(loop_i%300==0):
            print("At sentence: {}".format(-loop_i))
            print(training_data_with_other_stuff[-loop_i])
        loop_i-=1
    
    
    if(save_to_file):
        opt_info = "Normal_"
        if(include_sense):
            opt_info='WITH_SENSE_'
        
        processed_normal_csv_path = "{}/{}_absa_seq_labelled.csv".format(save_path, opt_info)
        with open(processed_normal_csv_path,'wb') as fout:
            csv_out = csv.writer(fout)
            csv_out.writerow(["Sentence","Sequence"])
            for row in training_data:
                csv_out.writerow(row)
                
        processed_additional_info_csv_path = "{}/{}_additional_info_seq_labelled.csv".format(save_path, opt_info)
        with open(processed_additional_info_csv_path, 'wb') as fout:
            csv_out = csv.writer(fout)
            csv_out.writerow(out_row)
            for row in training_data_with_other_stuff:
                csv_out.writerow(row) 
        
        processed_norm_training_data_pickle = "{}/{}_normal_training_list.pickle".format(save_path, opt_info)
        with open(processed_norm_training_data_pickle,'wb') as pickle_o:
            pickle.dump(training_data, pickle_o)
        
        processed_add_training_data_pickle = "{}/{}_additional_training_list.pickle".format(save_path, opt_info)
        with open(processed_add_training_data_pickle,'wb') as pickle_o:
            pickle.dump(training_data_with_other_stuff, pickle_o)
        
        processed_vocab_pickle = "{}/{}_vocab.pickle".format(save_path, opt_info)
        with open(processed_vocab_pickle,'wb') as pickle_o: 
            pickle.dump(token_to_id, pickle_o)
        
        token_to_freq_pickle = "{}/{}_tokenfreq.pickle".format(save_path, opt_info)
        with open(token_to_freq_pickle,'wb') as pickle_o: 
            pickle.dump(token_to_freq, pickle_o)
        
        tag_to_id_pickle = "{}/{}_tag2id.pickle".format(save_path, opt_info)
        with open(tag_to_id_pickle, 'wb') as pickle_o:
            pickle.dump(tag_to_id, pickle_o)
    
    return save_path, processed_normal_csv_path, processed_vocab_pickle 

class Domain:
    def __init__(self, name, primary_dir, data_path, raw_csv_file=None, already_processed = True):
        """
        primary_dir refers to the directory where the domain info is stored
        Ideally data_dir will be uniform throughout all domains.
        """
        self.name = name
        self.primary_dir = primary_dir
        self.raw_csv_file = raw_csv_file
        self.data_path = data_path #this is the processed data is stored
        self.vocab_path = None
        self.embedding_dir = None
        if(not already_processed):
            self.tr_data_path, self.vocab_path = self.run_data_processing(True) #with sense, lower
            _, _ = self.run_data_processing(False)
            
    def run_data_processing(self, with_sense= False):
        pd_file = pd.read_csv(self.raw_csv_file) #convert to pandas
        print("Processing raw data first with and then without sense")
        _, tr_data_path, vocab_path = store_dataset_info(pd_file, True, self.data_path, with_sense)
        return tr_data_path, vocab_path
        
    def get_domain_independent_features(self):
        None
        
    def get_feature_label_mutual_info(self):
        None
        
    def features(self):
        None
        
    def save_as_pickle(self,path):
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL) 
    
    
d2 = Domain('Rest','./Final_data/Domains/Rest/','./Final_data/Domains/Rest/','./Final_data/Semeval_14_ver1/Combined_restaurant.csv',True)

#with open('./Final_data/Domains/rest.pkl', 'wb') as output:
 #   pickle.dump(d2, output, pickle.HIGHEST_PROTOCOL)
