

#ifndef _LENET1_
#define _LENET1_

#define LENGTH_KERNEL0 5
#define LENGTH_KERNEL1 4

#define CENTER	(LENGTH_KERNEL0>>1)


#define LENGTH_FEATURE0	28										//28
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL0 + 1)	//24
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)					//12
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL0 + 1)  //8
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)					//4	
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL1 + 1)  //1

#define C1	4
#define C2	3
#define C3	10

#define LAYER1  4
#define LAYER2  4
#define LAYER3  12
#define LAYER4  12
#define OUTPUT  10

#define BIAS	0.0f



typedef struct LeNet1
{
	//C1 filters
	float filters1 [C1][LENGTH_KERNEL0*LENGTH_KERNEL0];
	//C2 filters
	float filters2 [C2][LENGTH_KERNEL0*LENGTH_KERNEL0];
	//C3 filters
	float filters3 [C3][LENGTH_KERNEL1*LENGTH_KERNEL1];
}LeNet1;

typedef struct Feature
{
	//input feature (image)
	float image[LENGTH_FEATURE0*LENGTH_FEATURE0];
	//layer1
	float layer1[LAYER1][LENGTH_FEATURE1*LENGTH_FEATURE1];
	//layer2
	float layer2[LAYER2][LENGTH_FEATURE2*LENGTH_FEATURE2];
	//layer3
	float layer3[LAYER3][LENGTH_FEATURE3*LENGTH_FEATURE3];
	//layer4
	float layer4[LAYER4][LENGTH_FEATURE4*LENGTH_FEATURE4];
	//layer5(output)
	float layer5[OUTPUT][LENGTH_FEATURE5];
}Feature;


#endif
