package cs446.weka.classifiers.trees;


	import java.io.File;
	import java.io.FileReader;

	import weka.classifiers.Evaluation;
	import weka.core.Instances;
	import weka.core.Instance;
	import cs446.weka.classifiers.trees.Id3;

	public class Sgd2 {

	    public static void main(String[] args) throws Exception {

		double wVector[] = new double[270];


		if (args.length != 1) {
		    System.err.println("Usage: test train alpha convergence_constant");
		    System.exit(-1);
		}

		Instances trainData = new Instances(new FileReader(new File(args[0])));
		Instances testData = new Instances(new FileReader(new File(args[0])));
		double alphaConstant = 0.01; //Passed in on command line, calculated in delta W later
		double convergeConstant = 0.01;

		trainData.setClassIndex(trainData.numAttributes() - 1);
		testData.setClassIndex(testData.numAttributes() - 1);

		Instances train = trainData;
		Instances test = testData;

		//Initialize weight vector to 0's
		for (int j = 0; j < 270; j++)
		{
			wVector[j] = 0;
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


