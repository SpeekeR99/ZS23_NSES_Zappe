#include "graphics/Visualization.h"

/**
 * Main function
 * @return Exit code
 */
int main() {
    /* Create visualization, which serves as the true main */
    Visualization visualization = Visualization();
    /* Loop until the user closes the window */
    visualization.run();

    return EXIT_SUCCESS;
}
