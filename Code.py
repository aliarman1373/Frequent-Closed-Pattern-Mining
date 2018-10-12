#Import Libraries
import nltk.stem
import collections as cl
from sklearn.feature_extraction.text import CountVectorizer

#Node class for the Creating Tree and N-List
class Node:
    def __init__(self, value=0):
        self.value = value
        self.childs = []
        self.count=1
        self.pre=0
        self.post=0
        
#Tree Class       
class Tree:
    def __init__(self):
        self.root = Node(0)
               
    def addToTree(self,r,words):
        '''
        Add a set of words to FP_Growth tree

        @params
        r: The root of tree
        words: A list of words for adding to tree
        '''
        node=Node(words[0])
        if(r.childs!=None):
            find_flag=False
            for ch in r.childs:
                try:
                    if(ch.value==node.value):
                        ch.count+=1
                        r=ch
                        find_flag=True
                except :
                    pass
                
            
            if find_flag==False:
                r.childs.append(node)
                r=node
        else:
            r.childs.append(node)
            r=node

        words.pop(0)
        if(len(words)>0):
            self.addToTree(r,words)
        
    def postorder(self,root,stack=[]):
        '''
        Treverse the tree in post-order and set value for each node post property 

        
        @params
        root: The root of tree
        stack: nodes would be pushed to the stack based on the post_order treversing
        '''
        
        for i,c in enumerate(root.childs):      
            self.postorder(c,stack) 
        stack.append(root) 
        root.post=len(stack)-1
 

   
    # A function to do postorder tree traversal 
    def preorder(self,root,stack=[]): 
        '''
        Treverse the tree in pre-order and set value for each node pre property 

        @params
        root: The root of tree
        stack: nodes would be pushed to the stack based on the pre_order treversing
        '''
        stack.append(root)
        root.pre=len(stack)-1
        for i,c in enumerate(root.childs):        
            self.preorder(c,stack)                         
        

    def getNodesByValues(self,root,val,tupleList=[]):
        '''
        Return all nodes of tree by specific value

        @params
        root: The root of tree
        val: a word
        tupleList: list of (pre,post,frequency) of each node which has 'val' value
        '''
        if root.value==val:
            nodeTuple=(root.pre,root.post,root.count)
            tupleList.append(nodeTuple)
        if(root.childs!=None):
            for i,c in enumerate(root.childs):
                self.getNodesByValues(c,val,tupleList)
        return tupleList 

#Add  stemming function to CountVectorizer
class StemmedCountVectorizer(CountVectorizer):
    '''
    Add english stemmer to the CountVectorizer
    '''
    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])



def CheckNList(n1,n2):
    '''
    get n1(pre,post,frequency) , n2(pre,post,frequency) and check if the n1 is the subset of n2 or not

    @params
    n1: list of tuples of node 1 
    n2: list of tuples of node 2
    '''
    i=0
    j=0
    while j<len(n1) and i<len(n2):
        if n2[i][0]<=n1[j][0] and n2[i][1]>=n1[j][1]:
            j+=1
        else:
            i+=1
    if j==len(n1):
        return True
    else:
        return False
   

def CheckPartNList(n1,n2):
    '''
    get n1(pre,post,frequency) , n2(pre,post,frequency) and check if the part of n1 is the subset of n2 or not

    @params
    n1: list of tuples of node 1 
    n2: list of tuples of node 2
    '''
    i=0
    j=0
    while j<len(n1) and i<len(n2):
        if n2[i][0]<=n1[j][0] and n2[i][1]>=n1[j][1]:
            j+=1
        else:
            i+=1
    if j>0:
        return True
    else:
        return False

def GetFrequency(n):
    '''
    calculate the frequency of given node and return an integer

    @params
    n: list of tuples of node  
    '''
    total=0
    for i,f in enumerate(n):
        total+=f[2]
    return total

def Merge(n_list,itemset_one,itemset_two):
    '''
    merge 2 itemset by thgeir value and return merged itemset

    @params
    itemset_one:   
    itemset_two: 
    '''
    temp_key=tuple(set(itemset_two+itemset_one))
    temp_value=n_list[itemset_one]
    key_value={temp_key:temp_value}
    return key_value

def CreateChild(n_list,itemset_one,itemset_two,minsupp):
    '''
    Create a child of 2 itemsets which a number of tuples of itemset1 is a subset of itemset2.
    after creating the child, the frequency of that would be checked and if it is greater-than or equal the child would be retuerned

    @params
    itemset_one:   
    itemset_two: 
    minsupp: minimum support value 
    '''

    list_child=[]
    dic={}
    n1=n_list[itemset_one]
    n2=n_list[itemset_two]
    frequency=0
    temp_c=[]
    i=0
    j=0
    while j<len(n1) and i<len(n2):
        if n2[i][0]<n1[j][0] and n2[i][1]>n1[j][1]:
            frequency+=n1[j][2]
            j+=1
        else:
            if(frequency>0 and frequency>=minsupp):
                value=[(n2[i][0],n2[i][1],frequency)]
                key=tuple(set(itemset_two+itemset_one))
                dic[key]=value
                list_child.append(dic)
            else:
                i+=1
            frequency=0
    
    return list_child

def SubsumptionCheck(fci_next,fci):
    '''
    Check if an itemset(fci) is already in fci_list or being subset of at least one element in the fci_list

    @params
    fci:a frequent closed itemset
    fci_next:List of frequent closed itemset
    '''
    for itemset_one,value_one in fci.items():
        for itemset_two,value_two in fci_next.items():
            if(CheckNList(value_one,value_two)==True and GetFrequency(value_one)==GetFrequency(value_two)):
                for word in itemset_one:
                    if word in itemset_two:
                        pass
                    else:
                        return False
                return True
    return False

def find_FCI(n_list,minsupp):
    '''
    get a list of frequent items of a text and the minimum support and return teh list of frequent closed itemsets

    @params
    n_list:list of frequent itemset
    minsupp:the minimum support
    '''
    fci_next={}
    items_list=list(n_list)
    for i in range(len(n_list)-1,0,-1):
        try:
            itemset_one=items_list[i]
        except :
            continue
        for j in range(i-1,-1,-1):    
            itemset_two=items_list[j]
            if  CheckNList(n_list[itemset_one],n_list[itemset_two]):
                if(GetFrequency(n_list[itemset_one])==GetFrequency(n_list[itemset_two])):
                    merged_items=Merge(n_list,itemset_one,itemset_two)
                    if(SubsumptionCheck(fci_next,merged_items)==False):
                        fci_next.update(merged_items)
                    try:
                        
                        del items_list[i]
                        del items_list[j]
                    except :
                        pass
                    
                else:
                    merged_items=Merge(n_list,itemset_one,itemset_two)
                    if(SubsumptionCheck(fci_next,merged_items)==False):
                        fci_next.update(merged_items)
                    try:
                        del items_list[i]               
                    except :
                        pass
            elif CheckPartNList(n_list[itemset_one],n_list[itemset_two]):
                childs=CreateChild(n_list,itemset_one,itemset_two,minsupp)
                if(len(childs)>0):
                    for ch in childs:
                        if(SubsumptionCheck(fci_next,ch)==False):
                            fci_next.update(ch)
    
    for item in reversed(items_list):
        if(SubsumptionCheck(fci_next,{item:n_list[item]})==False):
            temp={item:n_list[item]}
            fci_next.update(temp)

    if(len(fci_next)!=len(n_list)):
        temp=[]
        temp_dic={}
        for x in fci_next:
            temp.append(x)
        temp=temp[::-1]
        for obj in temp:
            temp_dic[obj]=fci_next[obj]
        return find_FCI(temp_dic,minsupp)
    else:
        return n_list

        
# create the transformer
vectorizer =StemmedCountVectorizer(encoding='utf-8', 
                decode_error='strict',
                strip_accents=None,
                lowercase=True,
                preprocessor=None, 
                tokenizer=None, 
                stop_words='english', 
                ngram_range=(1, 1), 
                analyzer='word', 
                max_df=1.0, 
                min_df=1, 
                max_features=None, 
                vocabulary=None, 
                binary=False)

#input the database
file_adress="2.txt"

#convert text file to string
file = open(file_adress, 'r')
text = file.read().strip()
file.close()

#split the text by paragraphs
y = [s.strip() for s in text.splitlines()]


#get the words of each paragraph and save it in 'paragraph_words'
paragraph_words=[]
for i,p in enumerate(y):
    try:
        vectorizer.fit([p])
        paragraph_words.append(vectorizer.get_feature_names())
    except ValueError:
        continue
    
#count each word in text
words_count = cl.Counter()
for sublist in paragraph_words:     
    words_count.update(sublist)

#normalize the frequency of words and delete the words with frequency less than support(0.1)
""" total = sum(words_count.values(), 0.0)
words_frequency={}
for key,value in words_count.items():
    words_count[key] = (words_count[key] /total)*100
    if words_count[key]>0.1:
        words_frequency[key]=words_count[key] """

# delete the words with frequency less than support(0.1)
total = sum(words_count.values(), 0.0)
words_frequency={}
for key,value in words_count.items():
    if (words_count[key] /total)*100>0.8:
        words_frequency[key]=words_count[key]
   


#sort the paragraph_words by the values and save that in a list
for i in range(len(paragraph_words)):
    temp={}
    for w in paragraph_words[i]:
        try:
            temp[w]=words_frequency[w]
        except:
            pass
    paragraph_words[i]=sorted(temp,key=temp.get,reverse=True)
  
#Add words in each paragraph which has the min supp and be sorted, to tree
tree=Tree()
temp_paragraph_words=paragraph_words
for pw in temp_paragraph_words:
    try:
        if(len(pw)>0):
            tree.addToTree(tree.root,pw)
    except (RuntimeError, TypeError, NameError):
        print(NameError)
    
tree.postorder(tree.root)
tree.preorder(tree.root)

#create the N-list which contatins itemset and the tuples(pre,post,frequency) of each itemset 
n_list={}
sorted_words=sorted(words_frequency,key=words_frequency.get,reverse=True)
for w in sorted_words:
    c_List=[]
    c_List=tree.getNodesByValues(tree.root,w,c_List)
    n_list[(w,)]=c_List


#find frequent closed itemsets and print them
result={}
result=find_FCI(n_list,4)
print(result.keys())










