struct tree{
	int key,value;
	struct tree *l,*r;
};
void free(void* p);
void insert(struct tree** root,struct tree* node){
	int nk = node->key;
	struct tree** pos = root;
	struct tree* now = *root;
	while(now != 0){
		if(now->key == nk){
			now->value = node->value;
			free(node);
			return;
		}
		if(now->value > nk){
			pos = &(now->l);
		} else { 
			pos = &(now->r);
		}
		now = *pos;
	}
	*pos = node;
}
