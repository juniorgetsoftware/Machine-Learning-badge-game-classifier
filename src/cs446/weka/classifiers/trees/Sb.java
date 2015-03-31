package cs446.weka.classifiers.trees;

import java.io.File;
import java.io.FileReader;

import java.lang.System;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;
import cs446.weka.classifiers.trees.Id3;

public class Sb {

    public static void main(String[] args) throws Exception {

	if (args.length != 2) {
	    System.err.println("Usage: test train alpha convergence_constant");
	    System.exit(-1);
	}

	Instances trainData = new Instances(new FileReader(new File(args[0])));
	Instances testData = new Instances(new FileReader(new File(args[1])));
	double alphaConstant=0.01;
	double convergeConstant=0.01;

	//Define class labels
	trainData.setClassIndex(trainData.numAttributes() - 1);
	testData.setClassIndex(testData.numAttributes() - 1);

	Instances train = trainData;
	Instances test = testData;

	Id3 classifiers[] = new Id3[100];

	//Randomly sample classifiers, make 4-depth trees
	for (int j = 0; j < 100; j++)
	{
		Instances sample = randomSample(train);
		Id3 classifier = new Id3();
		classifier.setMaxDepth(4);
		classifier.buildClassifier(sample);
		classifiers[j] = classifier;
	}
	double wVector[] = new double[100];

	//Initialize weight vector
	for (int i = 0; i < 100; i++)
	{	
		wVector[i] = classifiers[i].classifyInstance(train.instance(i));
	}

	boolean converges = false;

	//Deep copy of weight vector - need to check against later
	double checkConvergeVector[] = setConvergeVector(wVector);
	
	for (int i = 0; i < train.numInstances(); i++)
	{
		if (converges) 
			break;
		double currInstance[] = train.instance(i).toDoubleArray();
		double fx = fOfx(wVector, currInstance);
		for (int j = 0; j < wVector.length; j++)
		{
			//Update weight vector
			wVector[j] += updatePart(alphaConstant, currInstance[train.instance(i).classIndex()] == 0.0 ? -1 : 1, fx, currInstance[j]); 
		}
		//Check converge every fifty iterations
		if (i % 50 == 0 && i > 0)
		{
			 double difference = vectorDifference(checkConvergeVector, wVector);
			 checkConvergeVector = setConvergeVector(wVector);
			 if (difference < convergeConstant) //Convergence constant 
				 converges = true;
		}

	}



	double correct = 0;
	double incorrect = 0;
	
	for (int q = 0; q < test.numInstances(); q++)
	{
		double testArray[] = test.instance(q).toDoubleArray();

		double dot = fOfx(wVector, testArray);
		
		//Calculate dot of test instance, see if above or below calculated weight vector
		double prediction = dot > 0 ? 1 : 0;

		//Iterate naively
		if (prediction == testArray[test.instance(q).classIndex()])
			correct++;
		else
			incorrect++;
	}

	System.out.println("Total accuracy: "+ (correct/(incorrect+correct)));

    }

    //Math functions!

    static double updatePart(double alpha, double y, double fx, double xd)
    {
	return alpha*(y - fx)*xd;
    }

    static int[] randomIndices (Instances set)
    {

	int indexArray[] = new int[(set.numInstances())/2];

	for (int i = 0; i < (set.numInstances())/2; i++)
	{
		Random randomizer = new Random();
		randomizer.setSeed(System.currentTimeMillis());
		boolean check = false;
		int index = randomizer.nextInt(set.numInstances());
		while (check == false)
		{
			if (checkIfPresent(index, indexArray))
			{
				index = randomizer.nextInt(set.numInstances());
			}
			else
				check = true;
		}

		indexArray[i] = index;
		
	}

	return indexArray;
    }

    static Instances randomSample (Instances set)
    {
	Instances toReturn = new Instances(set);	//Deep copy
	toReturn.delete();
	int indicies[] = randomIndices(set);
	for (int i = 0; i < indicies.length; i++)
	{
		toReturn.add(set.instance(indicies[i]));
	}
	return toReturn;

    }


    static boolean checkIfPresent (int index, int choiceArray[])
    {
	for (int i = 0; i < choiceArray.length; i++)
	{
		if (choiceArray[i] == index)
			return true;
	}
	
	return false;

    }
		
    static double[] setConvergeVector(double weight[])
    {
	double conv[] = new double[weight.length];
	for (int i = 0; i < weight.length; i++)
	{
		conv[i] = weight[i];
	}

	return conv;

    }

    static double vectorDifference(double arr1[], double arr2[])
    {
	    double differenceVector[] = new double[arr1.length];
	    for (int i = 0; i < arr1.length; i++)
	    {
		differenceVector[i] = arr1[i] - arr2[i];
	    }

	    return getMagnitude(differenceVector);

    }

    static double getMagnitude(double vector[])
    {
	double curr = 0;
	for (int i = 0; i < vector.length; i++)
	{
		curr += (vector[i]*vector[i]);
	}
		
	return Math.sqrt(curr);

    }

    static double fOfx(double weight[], double instanceVector[])
    {
	double toReturn = 0;
	for (int i = 0; i < weight.length; i++)
	{
		toReturn += (weight[i] * instanceVector[i]);
	}

	return toReturn;
    }
}


