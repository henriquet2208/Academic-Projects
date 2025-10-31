#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_CHAIRS 3  // Maximum chairs in the waiting room
#define NUM_BARBERS 3 // Number of barbers

pthread_mutex_t mutex;               // Mutex for controlling access to shared variables
pthread_cond_t cond_customers;       // Condition variable to signal barbers when a customer arrives
pthread_cond_t cond_barber_ready;    // Condition variable for customers to wait for a barber

int waiting_customers = 0; // Count of customers waiting
int free_barbers = NUM_BARBERS; // Number of free barbers

void *barber_thread(void *arg) {
    while (1) {
        pthread_mutex_lock(&mutex);
        
        // Wait for customers to arrive
        while (waiting_customers == 0) {
            printf("Barber %ld is sleeping.\n", (long)arg);
            pthread_cond_wait(&cond_customers, &mutex); // Wait until a customer arrives
        }

        // There's at least one customer waiting
        waiting_customers--;
        free_barbers++;
        printf("Barber %ld is cutting hair, %d waiting customers left.\n", (long)arg, waiting_customers);
        
        // Signal a customer to proceed with the haircut
        pthread_cond_signal(&cond_barber_ready);
        
        pthread_mutex_unlock(&mutex);
        
        // Simulate haircut time
        sleep(3);

        pthread_mutex_lock(&mutex);
        free_barbers--; // Update number of free barbers
        pthread_mutex_unlock(&mutex);
    }
}

void *customer_thread(void *arg) {
    pthread_mutex_lock(&mutex);

    if (waiting_customers < NUM_CHAIRS) {
        // There's space in the waiting room
        waiting_customers++;
        printf("Customer arrived, %d waiting customers.\n", waiting_customers);
        
        // Wake up barbers if needed
        pthread_cond_signal(&cond_customers);

        // Wait for a barber to be ready
        while (free_barbers == 0) {
            pthread_cond_wait(&cond_barber_ready, &mutex);
        }

        // Getting a haircut
        printf("Customer is getting a haircut.\n");
    } else {
        // No chairs available, customer leaves
        printf("Customer left, no chairs available.\n");
    }

    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main() {
    pthread_t barbers[NUM_BARBERS];

    // Initialize mutex and condition variables
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond_customers, NULL);
    pthread_cond_init(&cond_barber_ready, NULL);

    // Create barber threads
    for (long i = 0; i < NUM_BARBERS; i++) {
        pthread_create(&barbers[i], NULL, barber_thread, (void *)i);
    }

    // Main loop to continuously create customers
    while (1) {
        // Create a customer thread
        pthread_t customer_t;
        pthread_create(&customer_t, NULL, customer_thread, NULL);
        
        // Detach the customer thread so it cleans up after itself
        pthread_detach(customer_t);

        // Sleep for a random interval before creating the next customer
        sleep(rand() % 5 + 1); // Random time interval between 1 and 5 seconds
    }

    // This point will never be reached, but good practice in real programs
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond_customers);
    pthread_cond_destroy(&cond_barber_ready);

    return 0;
}
