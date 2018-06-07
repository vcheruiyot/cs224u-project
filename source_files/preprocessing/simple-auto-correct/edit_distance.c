#include "samples/prototypes.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <stdbool.h>

void initialize_matrix(int nrows, int ncols, int matrix[nrows][ncols]){
	int i, j;
	for (i = 0; i < nrows; i++){
		for(j = 0; j < ncols; j++){
			matrix[i][j] = 0;
		}
	}
}
int min(int *minimum_values){
	int min = minimum_values[0];
	for(int i = 1; i < 3; i++){
		if(minimum_values[i] < min) min = minimum_values[i];
	}
	return min;
	
}
int minimum_edit_distance(char *misspelled, char *from_dict){
	int n = strlen(misspelled);
	int m = strlen(from_dict);
	int matrix[n+1][m+1];
	initialize_matrix(n, m, matrix);
	for (int i = 1; i < n + 1; i++){
		matrix[i][0] = matrix[i-1][0] + 1;
	} 
	for(int i = 1; i < m + 1; i++){
		matrix[0][i] = matrix[0][i-1] + 1;
	}
	for(int i = 1; i < n + 1; i++){
		for(int j = 1; j < m + 1; j++){
			int minimum_values[3];
			minimum_values[0] = matrix[i - 1][j] + 1;
			minimum_values[1] = matrix[i][j-1] + 1;
			if(misspelled[i - 1] == from_dict[j - 1]){
				minimum_values[2] = matrix[i - 1][j - 1];
			}else{
				minimum_values[2] = matrix[i - 1][j - 1] + 2;
			}
			matrix[i][j] = min(minimum_values);
			int sum = m + n;
			if (matrix[i][j] > (sum - (sum >> 2))){
				return 1000;
			}
		}
	}
	return matrix[n][m];
}
void clean_up(char **dictionary, char **misspelled){
	for(int i = 0; dictionary[i]; i++){
		free(dictionary[i]);
	}
	for(int i = 0; misspelled[i]; i++){
		free(misspelled[i]);
	}
	free(misspelled);
	free(dictionary);
}
char *correct_word(char **dictionary, char *misspelled){
	int score = INT_MAX;
	int name = 0;
	for(int i = 0; dictionary[i]; i++){
		int min_score = minimum_edit_distance(misspelled, dictionary[i]);
		if(min_score <= score){
			score = min_score;
			name = i;
		}
	}
	score = INT_MAX;
	char *correct = dictionary[name];
	return correct;
}
char **getWordsInDict(char **dictionary, char c){
	size_t initial = 32;
	size_t cur = 0;
	char **words = malloc(sizeof(char *) * initial);
	for(size_t i = 0; dictionary[i] != NULL; i++){
		if(cur == initial){
			char **new_block = realloc(words, sizeof(char *) * (initial *= 2));
			assert(new_block != NULL);
			words = new_block;
		}
		char *word = dictionary[i];
		if(word[0] == c){
			words[cur] = word;
			cur++;
		}
	}
	words[cur] = NULL;
	return words;
}
char **getWordsFromTweet(char *tweet, char *words[]){
	int index = 0;
	char *token = strtok(tweet, " ");
	while(token != NULL){
		words[index] = token;
		token = strtok(NULL, " ");
		index++;
	}
	words[index] = NULL;
	return words;
}
char **read_file(FILE *stream){
	size_t initial = 32;
	char **lines = malloc(sizeof(char *) * initial);
	size_t cur = 0;
	char *line;
	while((line = read_line(stream)) && strlen(line)){
		if(cur == initial){
			char **new_block = realloc(lines, sizeof(char *) * (initial *= 2));
			assert(new_block != NULL);
			lines = new_block;
		}
		lines[cur] = line;
		cur++;
	}
	lines[cur] = NULL;
	fclose(stream);
	return lines;
}
int main() {
	FILE *fp = fopen("samples/dictionary.txt", "r");
	FILE *stream = fopen("samples/test", "r");
	char **dictionary = read_file(fp);
    char **tweets = read_file(stream);
    for(size_t i = 0; tweets[i] != NULL; i++){
    	char *tweet = tweets[i];
    	char *wordsInTweet[256];
    	char **allWordsInTweet = getWordsFromTweet(tweet, wordsInTweet);
    	for(int j = 0; allWordsInTweet[j] != NULL; j++){
    		char *tweetWord = allWordsInTweet[j];
    		char **startWithLetter = getWordsInDict(dictionary, tweetWord[0]);
    		int indexInTweet = 0;
    		while(true){
    			char *wordInTweet = correct_word(startWithLetter, &tweetWord[0]);
    			printf("%s\n", wordInTweet);
    			printf("%s %s\n", &tweetWord[0], wordInTweet);
    			indexInTweet += strlen(wordInTweet);
    			if(indexInTweet >= strlen(tweetWord))break;
    		}
    	}
   }
    //correct_word(dictionary, misspelled);
    return 0;
}
