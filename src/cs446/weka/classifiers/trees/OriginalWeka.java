package cs446.weka.classifiers.trees;


	import java.io.File;
	import java.io.FileReader;

	import weka.classifiers.Evaluation;
	import weka.core.Instances;
	import cs446.weka.classifiers.trees.Id3;

	/** CS 446:
	  * This is a sample testing script for training and evaluating a Badges game classifier.
	  * Review the main method, and look up Weka documentation to understand what each line does.
	  */
	public class OriginalWeka {

	    public static void main(String[] args) throws Exception {

	    // Check for valid argument (the address of a .arff file, perhaps produced by FeatureGenerator.java)
		if (args.length != 2) {
		    System.err.println("Usage: WekaTester arff-file");
		    System.exit(-1);
		}

		// Load the data
		//Instances data = new Instances(new FileReader(new File(args[0])));

		// The last attribute (index N-1) is the class label
		//data.setClassIndex(data.numAttributes() - 1);

		// Train on 80% of the data and test on 20% 
	    // NOTE:    
	    // For your real experiments, you will need to create the training and testing folds manually using the provided splits (fold1, fold2, etc.)
	    // Make sure that your training data and testing data do not overlap!
		//Instances train = data.trainCV(5,0);
		//Instances test = data.testCV(5, 0);
		 Instances train = new Instances(new FileReader(new File(args[0])));
		  Instances test = new Instances(new FileReader(new File(args[1])));
		  
		  train.setClassIndex(train.numAttributes() - 1);
		 test.setClassIndex(test.numAttributes() - 1);

		// Create a new ID3 classifier. This uses the modified one where you can
		// set the depth of the tree.
		Id3 classifier = new Id3();

		// An example depth. If this value is -1, then the tree is grown to full
		// depth.
		classifier.setMaxDepth(-1);

		// Train on training data (make sure it doesn't overlap with testing data!)
		classifier.buildClassifier(train);
		
		/*for(int i=0;i<test.numInstances();i++){
			
			double k=classifier.classifyInstance(test.instance(i));
			System.out.println(k);
		} */

		// Print the classfier to the console (see the toString() method in Id3.java)
		System.out.println(classifier);
		System.out.println();

		// Evaluate on the test set
		Evaluation evaluation = new Evaluation(test);
		evaluation.evaluateModel(classifier, test);
		System.out.println(evaluation.toSummaryString()); 

	    
	}


}
