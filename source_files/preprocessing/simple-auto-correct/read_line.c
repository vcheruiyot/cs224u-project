#include "samples/prototypes.h"
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <string.h>



/*
* This function reads a single line of a file. By peforming a single
* linear swoop, and using fgets to fetch batches of characters from 
* the file, the functions dynamically allocated more memory incase 
* the line from the file  is longer than the initial estimate. 
*/
char *read_line(FILE *stream) {
	size_t curr_buf_size = 32;
	char chr = ungetc(getc(stream), stream);
   	if(!stream || chr == EOF)return NULL;
	char * buf = (char * ) malloc(curr_buf_size);
 	buf[curr_buf_size - 1] = 1; //appends 1, and 0 as markers in order to detect how far we've read the file
  	buf[curr_buf_size - 2] = 0;
    	size_t nread = 0;

   	 while (1) {
		fgets(buf + nread, curr_buf_size - nread, stream);    
     		if (buf[curr_buf_size-1] || '\n' == buf[curr_buf_size-2]){
			break;
		}		                                            
		nread = curr_buf_size-1;           
      		buf = realloc(buf, curr_buf_size *= 2);       
      		buf[curr_buf_size-1] = 1;  
      		buf[curr_buf_size-2] = 0;                                                                                 
      	}
	if(buf[strlen(buf) - 1] == '\n')buf[strlen(buf) - 1] = '\0';                                                                    
	return buf;                                              
}

