class DictionaryAnalyzer():
    """
    Analyzes a dictionary and executes a specific action depending on the 
    type of data found.
    """ 

    def __init__(self, 
                 str_callback = None, 
                 #number_callback = None,
                 verbose = False):
        self.str_callback = str_callback
        #self.num_callback = number_callback,
        self.verbose = verbose
    
    def compare_dtypes(self, value):
        """A function that compares the type of variable found in the dictionary
        """
        if(self.verbose): print(str(type(value)),':', value)
        
        # Number
        if type(value) == type(int()) or type(value) == type(float()):
            pass
            ## TODO: Enable two callbacks in the constructor. Right now the result of callable() for int is False.
            #if(callable(self.num_callback)):
            #    self.num_callback(value)
        # Dictionary
        elif type(value) == type(dict()):
            self.iterate_dict(value)
        # List
        elif type(value) == type(list()):
            self.iterate_list(value)
        # String: This is the last one because virtually anything can be casted to string
        elif type(value) == type(str()):
            if(callable(self.str_callback)): #parameter is a function
                self.str_callback(value)

    def iterate_list(self, value):
        for val in value:
            self.compare_dtypes(val)
    
    def iterate_dict(self, dictionary):
        for key,value in dictionary.items():
            self.current_key = key
            self.compare_dtypes(value)
    
    def analyze(self, dictionary):
        """Recursive exploration of a dictionary
        """
        self.iterate_dict(dictionary)