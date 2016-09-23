CC=g++ -std=c++11 
CXXFLAGS=-I.
OPENCV=`pkg-config --libs --cflags opencv` 
OPT=-O3
DEPS=photo_composer.hpp 
OBJ=main.o photo_composer.o 

%.o: %.cpp $(DEPS)
	$(CC) $(OPT)  -c -o $@ $< $(CFLAGS) `pkg-config --cflags opencv` 

photo: $(OBJ)
	$(CC) $(OPT)  -o $@ $^ $(CFLAGS) $(OPENCV)

.PHONY: clean

clean:
	rm -f *.o 
