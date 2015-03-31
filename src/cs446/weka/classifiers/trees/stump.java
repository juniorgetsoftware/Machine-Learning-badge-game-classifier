package cs446.weka.classifiers.trees;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.io.File;
import java.io.FileReader;


public class stump{
	
	static double [] weights =new double[100];


	public static void main(String[] args)	throws Exception{

		if (args.length != 2) {
		    System.err.println("Usage: test train alpha convergence_constant");
		    System.exit(-1);
		}

		
		Instances train = new Instances(new FileReader(new File(args[0])));
		Instances test = new Instances(new FileReader(new File(args[1])));
		
		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
	    Id3 classifiers[] = new Id3 [100];
	    
	    
	
	for (int j = 0; j < 100; j++)
	{
		Instances sample = randomSample(train);
		Id3 classifier = new Id3();
		classifier.setMaxDepth(4);
		classifier.buildClassifier(sample);
		classifiers[j] = classifier; 
	}
	
	Instances traindata=initializeAttributes(train);
	for(int t=0;t<=train.numInstances()-1;t++){
		
	Instance instance =makeInstance(train,traindata,classifiers,t);
	traindata.add(instance);
	}
	
	Instances testdata=initializeAttributes(test);
	for(int t=0;t<=test.numInstances()-1;t++){
		
	Instance instance =makeInstance(test,testdata,classifiers,t);
	testdata.add(instance);
	}
	/*for(int t=0;t<=test.numInstances();t++){
	Instance instance2=	makeInstance(testdata,classifiers,t);
	testdata.add(instance2);
	
	}*/
	update(traindata,testdata);
	
	}
	static String[] features;
	   private static  Instances initializeAttributes(Instances train) {
		     
		     FastVector zeroOne;
		    FastVector labels=new FastVector(2);
		    features= new String[101];
		    for(int i=0;i<100;i++){
		    	features[i]="label"+i+"=";
		    }
		    features[100]="classlabel";
		    // For each feature template, create a feature name for each letter in the alphabet (a to z)
		    // Store these feature names in a temporary list, feat_temp
			
			List<String> feat_temp = new ArrayList<String>();
			for (String f : features) {
			     
				feat_temp.add(f );
			   
			    }

		                   
		    // Replace the list of feature template names with the list of feature names in feat_temp
			features = feat_temp.toArray(new String[feat_temp.size()]);

		    // Store binary feature values
			zeroOne = new FastVector(2);
			zeroOne.addElement("1");
			zeroOne.addElement("0");

		    // Store class labels
			labels = new FastVector(2);
			labels.addElement("+");
			labels.addElement("-");
		    
			String nameOfDataset = "Ba";

			Instances instances;

			FastVector attributes = new FastVector(features.length+1);
	
			for (String featureName : features) {
			    attributes.addElement(new Attribute(featureName, zeroOne));
			}
			Attribute classLabel = new Attribute("Class", 101);

			attributes.addElement(classLabel);

			instances = new Instances(nameOfDataset, attributes, train.numInstances());

		
			return instances;

		    }
	   private static Instance makeInstance(Instances instances,
			   Instances newInstances, Id3 classifiers[],int k) throws NoSupportForMissingValuesException{
		   Instance instance =instances.instance(k);
		   Instance newInstance = new Instance(101);
		   newInstance.setDataset(newInstances); 
		 
			    for(int i=0;i<=99;i++){
				   
				    String name = "label"+i+"=";
				    Attribute att = newInstances.attribute(name);
			        // If feature is active, value is "1"; if inactive, value is "0"
				    double featureLabel = classifiers[i].classifyInstance(instance);
				    //String featureLabel = String.valueOf(classifiers[i].classifyInstance(instance));
				    newInstance.setValue(att, featureLabel);
			    }
				
			//}
		    // Set the example's class label
		    //String label=String.valueOf(instance.classValue());
			   double label=instance.classValue();
		   // TODO: set this from instance
		   Attribute att = newInstances.attribute("classlabel");
		   newInstances.setClassIndex(100);
		   newInstance.setValue(att, label);
			// newInstance.setClassValue(label);
             return newInstance;
		
	   }
	//Initialize weight vector
	 static void update(Instances train,Instances test) throws Exception  {
	for (int i = 0; i < 100; i++)
	{	
		weights[i] =0.1;
		
	}
	int i=0;
	int k=0;
	double [] weightA=new double[100];
	double [] weightB=new double[100];
	//boolean judge=false;
	//Loop	
	
    while(true){
    	boolean  judge=false;
    	if(i>=train.numInstances()){
    		i=0; 
    	 }
    Instance kk= train.instance(i);
    double [] enen=kk.toDoubleArray();
    double result=dotProd(enen,weights);
   
    for(int t=0;t<100;t++){
    weightA[t]=weights[t];
    }
    
    updateClassifier(train.instance(i), result);
    
    //compare the weights before and after update
    for(int t=0;t<100;t++){
        weightB[t]=weights[t];
    }
      judge=convergence(weightA,weightB);
      if(k>=500){
    	  break;
      }
      i++;
      k++;
    System.out.println(i);
    }
    
	double correct = 0;
	double incorrect = 0;
	
	for (int q = 0; q < test.numInstances(); q++)
	{
		double testArray[] = test.instance(q).toDoubleArray();
		double[] testArray2=new double[100];
		for(int t=0;t<=99;t++){
        testArray2[t]=testArray[t];
		}
		double dot = dotProd(weights, testArray2);
		
		//Calculate dot of test instance, see if above or below calculated weight vector
		double prediction = dot > 0.5 ? 1 : 0;

		//Iterate naively
		if (prediction == testArray[test.instance(q).classIndex()])
			correct++;
		else
			incorrect++;
	}

	System.out.println("Total accuracy: "+ (correct/(incorrect+correct)));
}

  

    static protected void updateClassifier(Instance instance, double result)
		      throws Exception {
	double alpha=0.001;	
	double y=instance.classValue();
	double[] att =instance.toDoubleArray();
	int j=instance.numAttributes();
	for(int i=0;i<100;i++){
	weights[i]=weights[i]-alpha*(result-y)*att[i];
	
	}
	}
    
	static boolean convergence(double [] weightsA,double [] weightsB){
		double sum=0;
		for(int j=0;j<=weightsA.length-1;j++){
			sum=sum+Math.pow((weightsA[j]-weightsB[j]),2);
		}
		sum=Math.sqrt(sum);
		System.out.println(sum);
		if(sum<=0.00001){
			return true;		
		}
		else return false;
	}

	static int[] randomIndices (Instances problem)
    {

	int indexArray[] = new int[(problem.numInstances())/2];

	for (int i = 0; i < (problem.numInstances())/2; i++)
	{
		Random random = new Random();
		random.setSeed(System.currentTimeMillis());
		boolean check = false;
		int index = random.nextInt(problem.numInstances());
		while (check == false)
		{
			if (checkIfPresent(index, indexArray))
			{
				index = random.nextInt(problem.numInstances());
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


    static boolean checkIfPresent (int index, int aimArray[]){
	for (int i = 0; i < aimArray.length; i++)
	{
		if (aimArray[i] == index)
			return true;
	}
	
	return false;

    } 

	
  static double dotProd(double [] attribute, double[] weights)
     {
      double result = 0;
      for(int i=0;i<=weights.length-1;i++){
  	    result=result+weights[i]*attribute[i];
       }
      return (result);
     }
  


  
  
	
	
  }


