all: kalman

kalman: kalman.o
	gcc -o kalman kalman.o -lm 
kalman.o: main.c
	gcc -c main.c -lm -o kalman.o
clean:
	rm -f kalman *.o
