#include "operator.h"
#include <string.h>

#define MAX_REGISTERED_OPS 64

static const operator_registry_t* s_registry[MAX_REGISTERED_OPS];
static int s_registry_count = 0;

int operator_register(const operator_registry_t* op) {
    if (!op || s_registry_count >= MAX_REGISTERED_OPS) return -1;
    s_registry[s_registry_count++] = op;
    return 0;
}

const operator_registry_t* operator_find(const char* name) {
    for (int i = 0; i < s_registry_count; i++) {
        if (strcmp(s_registry[i]->name, name) == 0) {
            return s_registry[i];
        }
    }
    return NULL;
}
