/**
 * @file LeNet.cuh 
 * @brief LeNet-1 DATA STRUCTURE
 */

#ifndef _LENET1_
#define _LENET1_

/** Filter's size for LAYER1 (C1: convolutional Layer 1), LAYER3 (C2: convolutional Layer 3) */
#define LENGTH_KERNEL0 5

/** Filter's size for LAYER4 (C3: convolutional Layer 5)*/
#define LENGTH_KERNEL1 4

/** Filters are centered around the target pixel for the convolution*/
#define CENTER	(LENGTH_KERNEL0>>1)

/** LAYER0's Feature size, input image is a 28x28 pixels image*/
#define LENGTH_FEATURE0	28	

/** LAYER1 Features size*/								
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL0 + 1)

/** LAYER2 Features size*/
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)

/** LAYER3 Features size*/
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL0 + 1)

/** LAYER4 Features size*/
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)

/** LAYER5 Feature size, output layer*/
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL1 + 1)

/** LAYER1's size, number of features in the layer1*/
#define LAYER1  4

/** LAYER2's size, number of features in the layer2*/
#define LAYER2  4

/** LAYER3's size, number of features in the layer3*/
#define LAYER3  12

/** LAYER4's size, number of features in the layer4*/
#define LAYER4  12

/** LAYER5's size, number of features in the layer5*/
#define OUTPUT  10

/** number of convolutional filters used for layer 1*/
#define C1	4

/** number of convolutional filters used for layer 3*/
#define C2	3

/** number of convolutional filters used for layer 5*/
#define C3	10

/** bias not included among trainable parameters*/
#define BIAS	0.0f

/** TRAINABLE PARAMETERS */
typedef struct Weigths
{
	/** C1 filters */
	float filters1 [C1][LENGTH_KERNEL0*LENGTH_KERNEL0];
	/** C2 filters */
	float filters2 [C2][LENGTH_KERNEL0*LENGTH_KERNEL0];
	/** C3 filters */
	float filters3 [C3][LENGTH_KERNEL1*LENGTH_KERNEL1];
}Weigths;

/** COLLECTION OF ALL FEATURES OF ALL DIFFERENT LAYERS */
typedef struct Feature
{
	/** input feature (image) */
	float image[LENGTH_FEATURE0*LENGTH_FEATURE0];
	/** layer1 */
	float layer1[LAYER1][LENGTH_FEATURE1*LENGTH_FEATURE1];
	/** layer2 */
	float layer2[LAYER2][LENGTH_FEATURE2*LENGTH_FEATURE2];
	/** layer3 */
	float layer3[LAYER3][LENGTH_FEATURE3*LENGTH_FEATURE3];
	/** layer4 */
	float layer4[LAYER4][LENGTH_FEATURE4*LENGTH_FEATURE4];
	/** layer5(output) */
	float layer5[OUTPUT];
}Feature;


#endif
