import tensorflow as tf

class CNNV2(tf.keras.Model):

    def __init__(self,input_dim,output_size):
        super().__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        
        self.input_size = [-1,self.input_dim,self.input_dim,1]
        
        #1st layer
        #Param 1: Number of filters
        #Param 2 = Kernel matrix size
        self.conv1 = tf.keras.layers.Conv2D(32,5,padding = 'same',activation='relu',name='conv1',
                    input_shape = self.input_size ,data_format='channels_last')
        
        
        #Max pool layer 1
        self.maxpool1 = tf.keras.layers.MaxPool2D(strides=(2,2),name='pool1')
        
        #2nd layer with 64 filters
        self.conv2 = tf.keras.layers.Conv2D(64,5,padding = 'same',activation='relu',name='conv2'
        )
        
        #Max pool layer 2
        self.maxpool2 = tf.keras.layers.MaxPool2D(strides=2,name='pool2')
        
        #Dense fully connected layer
        self.dense = tf.keras.layers.Dense(
            1024, activation='relu', name='dense')
        
        #Apply dropout during training to avoid co-adaptation
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        
        #Activation function will be applied to this to get predictions
        self.logits = tf.keras.layers.Dense(self.output_size,name='logits')
        
        
    def call(self,inputs,training=False):
    
        #Forward run of the model
        
        
        reshaped_inputs = tf.reshape(inputs,self.input_size)

        conv1 = self.conv1(reshaped_inputs)

        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)

        maxpool2 = self.maxpool2(conv2)

        #Calculate height width channels from maxpool2 layer
        #This is required for creating the fully connected layer
        hwc = maxpool2.shape.as_list()[1:]

        #Calculate the flattened size
        flattened_size = hwc[0] * hwc[1] * hwc[2]

        #Reshape max pool 2 from NHWC format to 2D format
        #First argument batch size is -1 and can take the value required for the reshaping operation
        pool2_flat = tf.reshape(maxpool2, [-1, flattened_size])


        dense = self.dense(pool2_flat)

        if training:
            dropout = self.dropout(dense,training=training)
            logits = self.logits(dropout)
            return tf.nn.softmax(logits)
        else:
            logits = self.logits(dense)
            return tf.nn.softmax(logits)
            #return self.convertLogits(logits)


    def convertLogits(self,logits):
        softmax = tf.nn.softmax(logits,name='softmax')
        prediction = tf.math.argmax(softmax,axis=-1)
        return prediction


        

        
        
        
        
        
        
        
        
        
        
        
        

        

    