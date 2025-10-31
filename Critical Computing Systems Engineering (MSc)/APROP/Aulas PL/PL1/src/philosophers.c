#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_PHILOSOPHERS 5

pthread_mutex_t forks[NUM_PHILOSOPHERS];
pthread_t philosophers[NUM_PHILOSOPHERS];

void *philosopher_pattern(void *arg) {
    int id = *(int *)arg;

    while (1) {
        // Thinking
        printf("Philosopher %d is thinking.\n", id);
        usleep(rand() % 1000);  // Simulate thinking time

        // Hungry and trying to pick up forks
        printf("Philosopher %d is hungry.\n", id);

        // Pick up forks in a specific order to prevent deadlock
        if (id % 2 == 0) {
            pthread_mutex_lock(&forks[id]);             // Pick up right fork
            pthread_mutex_lock(&forks[(id + 1) % NUM_PHILOSOPHERS]);  // Pick up left fork
        } else {
            pthread_mutex_lock(&forks[(id + 1) % NUM_PHILOSOPHERS]);  // Pick up left fork
            pthread_mutex_lock(&forks[id]);             // Pick up right fork
        }

        // Eating
        printf("Philosopher %d is eating.\n", id);
        usleep(rand() % 1000);  // Simulate eating time

        // Put down the forks
        pthread_mutex_unlock(&forks[id]);               // Put down right fork
        pthread_mutex_unlock(&forks[(id + 1) % NUM_PHILOSOPHERS]);    // Put down left fork

        // Back to thinking
        printf("Philosopher %d finished eating and is now thinking.\n", id);
    }

    return NULL;
}

int main() {
    int philosopher_ids[NUM_PHILOSOPHERS];

    // Initialize the mutexes (forks)
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_mutex_init(&forks[i], NULL);
        philosopher_ids[i] = i;
    }

    // Create philosopher threads
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_create(&philosophers[i], NULL, philosopher_pattern, &philosopher_ids[i]);
    }

    // Wait for all philosopher threads to finish (they actually never do in this infinite loop)
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_join(philosophers[i], NULL);
    }

    // Destroy the mutexes (this will never be reached in this infinite loop case)
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_mutex_destroy(&forks[i]);
    }

    return 0;
}
