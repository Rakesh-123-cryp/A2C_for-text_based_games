class graph_node:
    def __init__(self,name : str):#, direction=None, from_node=None):
        #if direction is not None and from_node is not None:
        #    self.n_nodes = ((from_node,direction),)
        #else:
        self.n_nodes = ()
        self.objects = ()
        self.name = name
    
    def set_n_node(self,node_name : str,direction : str):
        self.n_nodes += ((graph_node(node_name),direction),)
        
    def add_objects(self,item_name : str or tuple):
        if item_name.isinstance(str):
            self.objects = item_name
        else:
            self.objects += item_name
            
    def _get_opp(self, direction: str):
        if direction.lower == "n":
            return "s"
        elif direction == "s":
            return "n"
        elif direction == "e":
            return "w"
        elif direction == "w":
            return "e"
        elif direction == "se":
            return "nw"
        elif direction == "sw":
            return "ne"
        elif direction == "nw":
            return "se"
        else:
            return "sw"