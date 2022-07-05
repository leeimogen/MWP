#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 


# In[2]:


import ktrain
from ktrain import text


# In[3]:


DATA_PATH = r'..\all_operations.csv' # Direct "all_operations.csv" here
NUM_WORDS = 50000
MAXLEN = 350
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_csv(DATA_PATH, 'problem',
                      label_columns = ["addition", "subtraction", "multiplication", "division"],
                      val_filepath=None, # 10% of data will be used for validation
                      max_features=NUM_WORDS, maxlen=MAXLEN,
                      ngram_range=3, random_state=42)


# In[4]:


model = text.text_classifier('nbsvm', (x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))


# In[7]:


learner.autofit(0.001)


# 
