package cs446.weka.classifiers.trees;
import java.io.File;
import java.io.FileReader;
import java.util.Random;

import cs446.weka.classifiers.trees.SGD;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

/** CS 446:
  * This is a sample testing script for training and evaluating a Badges game classifier.
  * Review the main method, and look up Weka documentation to understand what each line does.
  */
public class WekaTester {

    public static void main(String[] args) throws Exception {

    // Check for valid argument (the address of a .arff file, perhaps produced by FeatureGenerator.java)
	if (args.length != 2) {
	    System.err.println("Usage: WekaTester arff-file");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));

	//The last attribute (index N-1) is the class label
	data.setClassIndex(data.numAttributes() - 1);
	//for(int i=0;i<=data.numInstances()-1;i++){
	//	data.instance(i).setClassValue(0.0);}
	// Train on 80% of the data and test on 20% 
    // NOTE:    
    // For your real experiments, you will need to create the training and testing folds manually using the provided splits (fold1, fold2, etc.)
    // Make sure that your training data and testing data do not overlap!
	  Instances train = new Instances(new FileReader(new File(args[0])));
	  Instances test = new Instances(new FileReader(new File(args[1])));
	  
	  train.setClassIndex(train.numAttributes() - 1);
	  test.setClassIndex(test.numAttributes() - 1);
	// Instances train = data.trainCV(5,0);
	// Instances test = data.testCV(5, 1);

	// Create a new ID3 classifier. This uses the modified one where you can
	// set the depth of the tree.
	SGD classifier = new SGD();

	// An example depth. If this value is -1, then the tree is grown to full
	// depth.
	/*classifier.setMaxDepth(4);*/
	// Train on training data (make sure it doesn't overlap with testing data!)
	classifier.buildClassifier(train);
	
    int i=0;
    int k=0;
    boolean  judge=false;
    double [] weightA=new double[271];
    double [] weightB=new double[271];
    //Loop
    do{
    //classifier.classifyInstance(test.instance(i));
    	 if(i>=59){
    		i=0; 
    	 }
    Instance kk= train.instance(i);
   
    double [] enen=kk.toDoubleArray();
    double result=classifier.dotProd(enen,classifier.weights);
    
    System.out.println(result); 
    
    for(int t=0;t<=270;t++){
    weightA[t]=classifier.weights[t];
    }
    
    classifier.updateClassifier(train.instance(i), result);
    
    for(int t=0;t<=270;t++){
        weightB[t]=classifier.weights[t];
        }
  
    judge=classifier.convergence(weightA,weightB);
    i++;
    k++;
    System.out.println(i);
    }while(k<=100);
    /*double correcte=0;
    for(int u=0;u<=test.numInstances()-1;u++){
    	double correctnum=test.instance(u).classValue();
    double haha=classifier.classifyInstance(test.instance(u));
    System.out.println(haha);
    correcte+=haha;
    }
    //System.out.println(correcte); */
	
	
	double correct=0;
	double incorrect=0;
	for (int q = 0; q < test.numInstances(); q++)
	{
		double testArray[] = test.instance(q).toDoubleArray();

		double dot = classifier.dotProd(classifier.weights,testArray);
		
	
		double classification = dot > 0.5 ? 1 : 0;

	
		if (classification == testArray[test.instance(q).classIndex()])
			correct++;
		else
			incorrect++;
	}

	System.out.println("Total accuracy: "+ (correct/(incorrect+correct)));

    }

    }

	

