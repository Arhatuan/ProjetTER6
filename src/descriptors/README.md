# How to add a new descriptor

'***descriptors***' file :
* function(s) to compute the descriptor
* add support for the descriptor in the function '*image_processing*' (using whatever is needed from what you added in *DescriptorsParameters* and *DescriptorsData*)

'***Descriptor***' enumeration :
* add a value for your new descriptor

'***DescriptorsParameters***' :
* add any necessary attribute
* initialize the attribute (probably just a flag to indicate that this descriptor needs to be computed)

'***DescriptorsData***' :
* add an attribute that will contain the data for that descriptor (for one image)
* initialize the attribute (probably with a list)

'***ListDescriptorsData***' :
* add an attribute that will contain the list of this descriptor's data from several images
* initialize the attribute
* in the function '*add_descriptors_data_from_one_image*', add the data for the descriptor from a '*DescriptorsData*' instance, to the new attribute
* in the function '*generate_X_data_from_descriptors*', extend the variable 'sum_descriptors' with your descriptor's data if required
  
At the project's root, in the function '***parse_args***' :
* signal the new descriptor in the metavar of the argument for descriptors
* add the descriptor to the combinations list (for example, if it is intended to input a string 'testDescriptor' in the command line, it must be converted to a value of the *Descriptor* enumeration)