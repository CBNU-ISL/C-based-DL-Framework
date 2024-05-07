CC = gcc
CFLAGS = -Wall -Wextra
LDLIBS = -lm -lopenblas -lpthread -lgfortran

SRCS = main.c DL_module.c utils.c cJSON.c
OBJS = $(SRCS:.c=.o)
TARGET = DLTEST

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDLIBS)

main.o: main.c DL_module.h utils.h cJSON.h
	$(CC) $(CFLAGS) -c $< -o $@

DL_module.o: DL_module.c DL_module.h utils.h cJSON.h
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.c utils.h cJSON.h
	$(CC) $(CFLAGS) -c $< -o $@

cJSON.o: cJSON.c cJSON.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)