#!/usr/bin/env python3

names = [
('Devon', 'm'),
('Michael', 'm'),
('Jonathon', 'm'),
('Morten', 'm'),
('Christian', 'm'),
('Nicklas', 'm'),
('Peter', 'm'),
('Nanna', 'f'),
('Laura', 'f'),
('Angela', 'f'),
('Natalie', 'f'),
('Barbara', 'f'),
('Juanita', 'f'),
('Jennifer', 'f'),
('Eleanor', 'f')
]

def feature_extraction(word):
	return word[-1]    # The last letter of a word

def train(data):
	model = {} # Define the dictionary that are going to store the counts for each class.
	example_count = 0 # Count of how many examples that you have gone through.

	for example in data:
		X, y = example # X is the name, y is the gender.
		f_x = feature_extraction(X) # get the last letter
		
		#print(X,f_x,y) #
		#print(model)   # Uncomment these to see whats inside.
		#print()        #

		if y not in model: # If the gender is not in the model
			model[y] = {}  # Sub dictionary

		if f_x not in model[y]: # If that last letter was not in the model given that gender.
			model[y][f_x] = 0   # Initialize as 0

		model[y][f_x] = model[y][f_x] + 1  # Add 1

		example_count += 1 # Increment example count

		# keep track of how many times we see each class
		if 'observed' not in model[y]: # Count how many values that are observed given the label
			model[y]['observed'] = 0   #

		model[y]['observed'] = model[y]['observed'] + 1 # increment counter

	return model

parameters = train(names) # Count the names
print(parameters)		  # print the names

def predict(model, data):
	f_x = feature_extraction(data) # Get the last letter

	mle = {} # Create a dictionary for the maximum likelihood
	for label in model: # For each label (gender)
		mle[label] = 0  # Initialize as 0

		if f_x in model[label]: # If this last letter was seen in the training data
			mle[label] = model[label][f_x] / model[label]['observed'] # Get the maximum likelihood for that class.

	print(mle) # print the maximum likelihood of each class.

	best_estimate = ['', 0] # A container that are getting the class and maximum likelihood
	for label in mle:       # For classes we have calculated mle on
		if mle[label] > best_estimate[1]: # If the mle for this class is higher than the one before
			best_estimate = [label, mle[label]] # Set the value in the container

	return best_estimate

test_value = 'Jane'

prediction = predict(parameters, test_value)

print(prediction)
print('It looks like', test_value, 'is from class', prediction[0])