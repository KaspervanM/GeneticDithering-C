#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <ppm.h>
#include <string.h>

#include "pbm.h"
#include "pgm.h"

// --- Configuration constants ---
#define CHROMOSOME_SIZE (150 * 150)    // The number of genes per chromosome
#define POOL_SIZE 800          // The total number of chromosomes in our gene pool
#define GENERATIONS 100000       // How many iterations of the algorithm to run
#define MUTATION_RATE_CHANGE 0.9
#define CROSSOVER_RATE_CHANGE 0.05
#define CROSSOVER_DEGREE_CHANGE 0.97
#define HYPERPARAMETER_HISTORY 20

PgmImage *target;
constexpr double original_mutation_rate = 0.0005;
constexpr double original_crossover_rate = 0.75;
constexpr double original_crossover_degree = 0.35;
constexpr double original_hyperparameter_threshold = 0.0001;
constexpr double reset_multiplier = 0.1;
double hyperparameter_threshold_change = 0.75;
double mutation_rate = original_mutation_rate;
double crossover_rate = original_crossover_rate;
double crossover_degree = original_crossover_degree;
double hyperparameter_threshold = original_hyperparameter_threshold;
int reset_counter = 100;

// --- Individual Chromosome Structure ---
typedef struct {
    uint8_t genes[CHROMOSOME_SIZE];
    double loss;
} Chromosome;

// --- Typedefs for function pointers ---
typedef double (*LossFunction)(const Chromosome *chromosome);
typedef void (*MutationFunction)(Chromosome *chromosome);
typedef void (*CrossoverFunction)(const Chromosome *parent1, const Chromosome *parent2, Chromosome *child);


// --- faster rand

static uint32_t rand_state = 2463534242; // Or any non-zero seed

uint32_t fast_rand() {
    rand_state ^= rand_state << 13;
    rand_state ^= rand_state >> 17;
    rand_state ^= rand_state << 5;
    return rand_state;
}

double fast_rand01() {
    return (double)fast_rand() / (double)UINT32_MAX;
}

// --- Conversion from a Chromosome to a PgmImage
PgmImage* chromosomeToPgm(const Chromosome *chromosome) {
    // Check if the chromosome can be converted
    if (CHROMOSOME_SIZE != target->width_ * target->height_) {
        fprintf(stderr, "Error: could not convert chromosome to image, their sizes do not match up.");
        return nullptr;
        }

    // Allocate memory for image data
    PgmImage *image = AllocatePgm(target->width_, target->height_);

    // Copy chromosome pixel data to image
    memcpy(image->data_, chromosome->genes, CHROMOSOME_SIZE);
    return image;
}

double mse(const PgmImage *img1, const PgmImage *img2) {
    // Ensure that the images have the same dimensions.
    if (img1->width_ != img2->width_ || img1->height_ != img2->height_) {
        // In a production setting you might handle this error differently.
        fprintf(stderr, "Error: image dimensions do not match.\n");
        return -1.0;
    }

    const uint32_t total_pixels = img1->width_ * img1->height_;
    uint64_t sum_squared_error = 0;

    // Use pointer arithmetic for efficient data access.
    const uint8_t *p1 = img1->data_;
    const uint8_t *p2 = img2->data_;
    for (uint32_t i = 0; i < total_pixels; i++) {
        const int diff = (int)p1[i] - (int)p2[i];
        sum_squared_error += (uint64_t)(diff * diff);
    }

    // Compute and return the mean squared error.
    return (double)sum_squared_error / total_pixels;
}

// Global SSIM implementation for two PGM images.
double global_ssim(const PgmImage *img1, const PgmImage *img2) {
    // Check that the image dimensions are equal.
    if (img1->width_ != img2->width_ || img1->height_ != img2->height_) {
        fprintf(stderr, "Error: image dimensions do not match.\n");
        return -1.0;
    }

    const uint32_t total_pixels = img1->width_ * img1->height_;
    const uint8_t *p1 = img1->data_;
    const uint8_t *p2 = img2->data_;

    // First pass: Compute means (μ₁ and μ₂)
    double sum1 = 0.0, sum2 = 0.0;
    for (uint32_t i = 0; i < total_pixels; i++) {
        sum1 += (double)p1[i];
        sum2 += (double)p2[i];
    }
    double mu1 = sum1 / total_pixels;
    double mu2 = sum2 / total_pixels;

    // Second pass: Compute variances (σ₁², σ₂²) and covariance (σ₁₂)
    double var1 = 0.0, var2 = 0.0, covar = 0.0;
    for (uint32_t i = 0; i < total_pixels; i++) {
        double d1 = (double)p1[i] - mu1;
        double d2 = (double)p2[i] - mu2;
        var1 += d1 * d1;
        var2 += d2 * d2;
        covar += d1 * d2;
    }

    // Use (N-1) in the denominator if a sample variance is desired,
    // otherwise you can use N for the population variance.
    var1 /= (double)(total_pixels - 1);
    var2 /= (double)(total_pixels - 1);
    covar /= (double)(total_pixels - 1);

    // Constants for stability, based on the dynamic range of pixel values.
    // For an 8-bit image, L = 255.
    const double L = 255.0;
    const double K1 = 0.01, K2 = 0.03;
    const double C1 = (K1 * L) * (K1 * L);  // typically ≈ 6.5025
    const double C2 = (K2 * L) * (K2 * L);  // typically ≈ 58.5225

    // SSIM formula:
    // SSIM = ((2*μ₁*μ₂ + C1) * (2*σ₁₂ + C2)) / ((μ₁² + μ₂² + C1) * (σ₁² + σ₂² + C2))
    double numerator   = (2 * mu1 * mu2 + C1) * (2 * covar + C2);
    double denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (var1 + var2 + C2);

    if (denominator == 0.0) {
        fprintf(stderr, "Error: denominator in SSIM calculation is zero.\n");
        return -1.0;
    }

    return numerator / denominator;
}

// Compute SSIM over 8x8 blocks over the entire image
double ssim(const PgmImage *img1, const PgmImage *img2) {
    // Check that image dimensions match.
    if (img1->width_ != img2->width_ || img1->height_ != img2->height_) {
        fprintf(stderr, "Error: image dimensions do not match.\n");
        return -1.0;
    }

    // Constants as per the SSIM paper.
    const double K1 = 0.01;
    const double K2 = 0.03;
    const double L = 255.0; // Dynamic range of pixel values.
    const double C1 = (K1 * L) * (K1 * L);
    const double C2 = (K2 * L) * (K2 * L);

    constexpr int window_size = 8; // 8x8 blocks.
    double ssim_sum = 0.0;
    int count = 0;

    // Iterate over blocks including borders that may have smaller sizes.
    for (uint32_t i = 0; i < img1->height_; i += window_size) {
        for (uint32_t j = 0; j < img1->width_; j += window_size) {
            // Determine the actual block size
            uint32_t block_h = ((i + window_size) > img1->height_) ? (img1->height_ - i) : window_size;
            uint32_t block_w = ((j + window_size) > img1->width_) ? (img1->width_ - j) : window_size;
            int win_n = block_h * block_w;

            // Compute means
            double mu1 = 0.0, mu2 = 0.0;
            for (uint32_t r = 0; r < block_h; r++) {
                for (uint32_t c = 0; c < block_w; c++) {
                    uint32_t idx = (i + r) * img1->width_ + (j + c);
                    mu1 += (double)img1->data_[idx];
                    mu2 += (double)img2->data_[idx];
                }
            }
            mu1 /= win_n;
            mu2 /= win_n;

            // Compute variances and covariance
            double sigma1_sq = 0.0, sigma2_sq = 0.0, sigma12 = 0.0;
            for (uint32_t r = 0; r < block_h; r++) {
                for (uint32_t c = 0; c < block_w; c++) {
                    uint32_t idx = (i + r) * img1->width_ + (j + c);
                    double p1 = (double)img1->data_[idx];
                    double p2 = (double)img2->data_[idx];
                    sigma1_sq += (p1 - mu1) * (p1 - mu1);
                    sigma2_sq += (p2 - mu2) * (p2 - mu2);
                    sigma12   += (p1 - mu1) * (p2 - mu2);
                }
            }
            sigma1_sq /= (win_n > 1 ? (win_n - 1) : 1);
            sigma2_sq /= (win_n > 1 ? (win_n - 1) : 1);
            sigma12   /= (win_n > 1 ? (win_n - 1) : 1);

            // Compute the SSIM for this block (using the constants C1 and C2 defined earlier)
            double numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2);
            double denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2);
            double ssim_val = numerator / denominator;
            ssim_sum += ssim_val;
            count++;
        }
    }

    // Return the average SSIM index over all blocks.
    return (count > 0) ? ssim_sum / count : 0.0;
}

double ssim_overlapping(const PgmImage *img1, const PgmImage *img2) {
    // Check that image dimensions match.
    if (img1->width_ != img2->width_ || img1->height_ != img2->height_) {
        fprintf(stderr, "Error: image dimensions do not match.\n");
        return -1.0;
    }

    // Constants as per the SSIM paper.
    const double K1 = 0.01;
    const double K2 = 0.03;
    const double L = 255.0; // Dynamic range of pixel values.
    const double C1 = (K1 * L) * (K1 * L);
    const double C2 = (K2 * L) * (K2 * L);

    constexpr int window_size = 8; // 8x8 blocks.
    double ssim_sum = 0.0;
    int count = 0;

    const int stride = 4;
    for (uint32_t i = 0; i < img1->height_ - 1; i += stride) {
        for (uint32_t j = 0; j < img1->width_ - 1; j += stride) {
            // Define window boundaries: if the window goes beyond the image,
            // compute using available pixels (or pad them via mirroring, if preferred)
            uint32_t block_h = (i + window_size > img1->height_) ? (img1->height_ - i) : window_size;
            uint32_t block_w = (j + window_size > img1->width_) ? (img1->width_ - j) : window_size;
            int win_n = block_h * block_w;

            // Compute means
            double mu1 = 0.0, mu2 = 0.0;
            for (uint32_t r = 0; r < block_h; r++) {
                for (uint32_t c = 0; c < block_w; c++) {
                    uint32_t idx = (i + r) * img1->width_ + (j + c);
                    mu1 += (double)img1->data_[idx];
                    mu2 += (double)img2->data_[idx];
                }
            }
            mu1 /= win_n;
            mu2 /= win_n;

            // Compute variances and covariance
            double sigma1_sq = 0.0, sigma2_sq = 0.0, sigma12 = 0.0;
            for (uint32_t r = 0; r < block_h; r++) {
                for (uint32_t c = 0; c < block_w; c++) {
                    uint32_t idx = (i + r) * img1->width_ + (j + c);
                    double p1 = (double)img1->data_[idx];
                    double p2 = (double)img2->data_[idx];
                    sigma1_sq += (p1 - mu1) * (p1 - mu1);
                    sigma2_sq += (p2 - mu2) * (p2 - mu2);
                    sigma12   += (p1 - mu1) * (p2 - mu2);
                }
            }
            sigma1_sq /= (win_n > 1 ? (win_n - 1) : 1);
            sigma2_sq /= (win_n > 1 ? (win_n - 1) : 1);
            sigma12   /= (win_n > 1 ? (win_n - 1) : 1);

            // Compute the SSIM for this block (using the constants C1 and C2 defined earlier)
            double numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2);
            double denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2);
            double ssim_val = numerator / denominator;
            ssim_sum += ssim_val;
            count++;
        }
    }

    // Return the average SSIM index over all blocks.
    return (count > 0) ? ssim_sum / count : 0.0;
}

// Loss function using SSIM.
// Lower loss is better, so we define loss = 1 - SSIM
double ssim_loss_function(const Chromosome *chromosome) {
    // Convert the chromosome to a PgmImage.
    PgmImage *pgm = chromosomeToPgm(chromosome);

    // Compute the SSIM index with the target image.
    double ssim_index = ssim(pgm, target);

    // Free the generated image.
    free(pgm->data_);
    free(pgm);

    // Return loss (0 loss for identical images).
    return (1.0 - ssim_index) * (1.0 - ssim_index); // Squared to make it more sensitive to small changes.
}

double adaptive_local_ssim(const PgmImage *img1, const PgmImage *img2,
                             uint32_t i, uint32_t j, uint32_t window_size,
                             double threshold, double low_weight) {
    double mu1 = 0.0, mu2 = 0.0;
    uint32_t win_n = window_size * window_size;

    // Compute means over the window.
    for (uint32_t r = 0; r < window_size; r++) {
        for (uint32_t c = 0; c < window_size; c++) {
            uint32_t idx = (i + r) * img1->width_ + (j + c);
            mu1 += (double)img1->data_[idx];
            mu2 += (double)img2->data_[idx];
        }
    }
    mu1 /= win_n;
    mu2 /= win_n;

    // Compute variances to gauge contrast.
    double sigma1_sq = 0.0, sigma2_sq = 0.0, sigma12 = 0.0;
    for (uint32_t r = 0; r < window_size; r++) {
        for (uint32_t c = 0; c < window_size; c++) {
            uint32_t idx = (i + r) * img1->width_ + (j + c);
            double pix1 = (double)img1->data_[idx];
            double pix2 = (double)img2->data_[idx];
            sigma1_sq += (pix1 - mu1) * (pix1 - mu1);
            sigma2_sq += (pix2 - mu2) * (pix2 - mu2);
            sigma12   += (pix1 - mu1) * (pix2 - mu2);
        }
    }
    sigma1_sq /= (win_n - 1);
    sigma2_sq /= (win_n - 1);
    sigma12   /= (win_n - 1);

    // Compute the raw SSIM value for this window.
    const double K1 = 0.01;
    const double K2 = 0.03;
    const double L = 255.0;
    const double C1 = (K1 * L) * (K1 * L);
    const double C2 = (K2 * L) * (K2 * L);
    double numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2);
    double denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2);
    double ssim_val = numerator / denominator;

    double local_variance = sigma2_sq;

    // Determine a weight based on the local variance.
    // If variance is too low, reduce the influence of the local window.
    double weight = (local_variance < threshold) ? low_weight : 1.0;

    return weight * ssim_val;
}

double variance_of_image(PgmImage *pgm) {
    double sum = 0.0;
    uint32_t count = pgm->width_ * pgm->height_;
    for (uint32_t i = 0; i < pgm->height_; i++) {
        for (uint32_t j = 0; j < pgm->width_; j++) {
            sum += (double)pgm->data_[i * pgm->width_ + j];
        }
    }
    double mean = sum / count;
    double variance = 0.0;
    for (uint32_t i = 0; i < pgm->height_; i++) {
        for (uint32_t j = 0; j < pgm->width_; j++) {
            double diff = (double)pgm->data_[i * pgm->width_ + j] - mean;
            variance += diff * diff;
        }
    }
    variance /= count;
    return variance;
}

double adaptive_ssim_loss_function(const Chromosome *chromosome) {
    PgmImage *pgm = chromosomeToPgm(chromosome);

    // Adaptive local SSIM.
    const uint32_t window_size = 8;
    const double global_variance = variance_of_image(target);
    const double variance_threshold = 0.05 * global_variance;  // Experimentally determined.
    const double low_weight = 0.05;           // Down-weight uniform regions.
    double local_ssim_sum = 0.0;
    int count = 0;

    // Use a stride (you may choose to make it overlapping or not).
    const uint32_t stride = 4;
    for (uint32_t i = 0; i <= pgm->height_ - window_size; i += stride) {
        for (uint32_t j = 0; j <= pgm->width_ - window_size; j += stride) {
            local_ssim_sum += adaptive_local_ssim(pgm, target, i, j, window_size, variance_threshold, low_weight);
            count++;
        }
    }

    double avg_local_ssim = (count > 0) ? local_ssim_sum / count : 0.0;

    // Define the loss as 1 - combined SSIM (0 loss is perfect).
    double loss = (1.0 - avg_local_ssim) * (1.0 - avg_local_ssim); // Squared to make it more sensitive to small changes.

    free(pgm->data_);
    free(pgm);
    return loss;
}

double ssim_overlapping_loss_function(const Chromosome *chromosome) {
    // Convert the chromosome to a PgmImage.
    PgmImage *pgm = chromosomeToPgm(chromosome);

    // Compute the SSIM index with the target image.
    double ssim_index = ssim_overlapping(pgm, target);

    // Free the generated image.
    free(pgm->data_);
    free(pgm);

    // Return loss (0 loss for identical images).
    return (1.0 - ssim_index) * (1.0 - ssim_index); // Squared to make it more sensitive to small changes.
}


double ssim_overlapping_and_global(const Chromosome *chromosome) {
    PgmImage *pgm = chromosomeToPgm(chromosome);
    const double global_loss = global_ssim(pgm, target);
    const double local_loss = ssim_overlapping(pgm, target);

    free(pgm->data_);
    free(pgm);
    return (1 - global_loss) * (1 - global_loss) + local_loss;
}

double ssim_adaptive_and_global_loss_function(const Chromosome *chromosome) {
    PgmImage *pgm = chromosomeToPgm(chromosome);
    const double global_loss = global_ssim(pgm, target);
    const double local_loss = adaptive_ssim_loss_function(chromosome);

    free(pgm->data_);
    free(pgm);
    return (1 - global_loss) * (1 - global_loss) + 2 * local_loss;
}

double mse_loss_function(const Chromosome *chromosome) {
    PgmImage *pgm = chromosomeToPgm(chromosome);
    const double loss = mse(pgm, target);
    free(pgm->data_);
    free(pgm);
    return loss;
}

double global_ssim_loss_function(const Chromosome *chromosome) {
    PgmImage *pgm = chromosomeToPgm(chromosome);
    const double loss = global_ssim(pgm, target);
    free(pgm->data_);
    free(pgm);
    return (1 - loss) * (1 - loss); // SSIM is between 0 and 1, so we want to maximize it. squaring it makes it more sensitive to small changes.
}

// --- Example Mutation Function ---
// This mutation function randomly changes each gene with a probability defined by MUTATION_RATE.
void mutation_function(Chromosome *chromosome) {
    const int nr_of_genes_to_mutate = (int)(CHROMOSOME_SIZE * mutation_rate);
    for (int i = 0; i < nr_of_genes_to_mutate; i++) {
        const int index = fast_rand() % CHROMOSOME_SIZE;
        // Flip the bit at the index
        chromosome->genes[index] = (chromosome->genes[index] == 0) ? 255 : 0;
    }
}

// --- Example Crossover Function ---
// This crossover function performs single-point crossover: it picks a random crossover point and combines parent genes.
void crossover_function(const Chromosome *parent1, const Chromosome *parent2, Chromosome *child) {
    memcpy(child->genes, parent1->genes, CHROMOSOME_SIZE);
    if (fast_rand01() > crossover_rate) {
        return;
    }
    const int nr_of_genes_from_parent2 = (int)(CHROMOSOME_SIZE * crossover_degree);
    for (int i = 0; i < nr_of_genes_from_parent2; i++) {
        const int index = fast_rand() % CHROMOSOME_SIZE;
        child->genes[index] = parent2->genes[index];
    }
}

// --- Initialize Population ---
Chromosome * initialize_population(const LossFunction lossFunc) {
    Chromosome *population = malloc(sizeof(Chromosome) * POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; i++) {
        for (int j = 0; j < CHROMOSOME_SIZE; j++) {
            population[i].genes[j] = fast_rand() % 2 * 255;
        }
        population[i].loss = lossFunc(&population[i]);
    }
    return population;
}

// --- Initialize Population with images that are thresholds of the target image, but each mutated a little bit ---
Chromosome * initialize_population_of_threshold(const LossFunction lossFunc) {
    Chromosome *population = malloc(sizeof(Chromosome) * POOL_SIZE);
    const double old_mutation_rate = mutation_rate;
    mutation_rate = 0.1;
    for (int i = 0; i < POOL_SIZE; i++) {
        PbmImage *pbm = PgmToPbm(target, MiddleThreshold);
        PgmImage *pgm = PbmToPgm(pbm);
        for (int j = 0; j < CHROMOSOME_SIZE; j++) {
            population[i].genes[j] = pgm->data_[j];
        }
        // mutate to create diversity
        mutation_function(&population[i]);
        population[i].loss = lossFunc(&population[i]);
        free(pbm->data_);
        free(pbm);
        free(pgm->data_);
        free(pgm);
    }
    mutation_rate = old_mutation_rate;
    return population;
}

// Function to perform weighted (roulette wheel) selection.
Chromosome* roulette_selection(Chromosome population[POOL_SIZE]) {
    double total_fitness = 0.0;
    double fitness[POOL_SIZE]; // Array to hold computed fitness values

    double min_loss = population[0].loss;
    double max_loss = population[0].loss;
    for (int i = 1; i < POOL_SIZE; i++) {
        if (population[i].loss < min_loss) {
            min_loss = population[i].loss;
        }
        if (population[i].loss > max_loss) {
            max_loss = population[i].loss;
        }
    }

    // Compute the fitness for each chromosome: lower loss gives higher fitness.
    for (int i = 0; i < POOL_SIZE; i++) {
        fitness[i] = max_loss - population[i].loss;
        total_fitness += fitness[i];
    }

    // All equal probabilities if the population is too similar.
    if (total_fitness < 1e-50) {
        return nullptr;
    }
    // Generate a random number between 0 and total_fitness.
    double r = fast_rand01() * total_fitness;
    // printf("r: %f, ", r);

    // Select a chromosome based on weighted fitness.
    int i = -1;
    while (r > 0) {
        i ++;
        i %= POOL_SIZE;
        r -= fitness[i];
    }
    // printf("fitness of selected: %f\n", fitness[i]);
    return &population[i];
}

// --- Select Two Parents (Roulette Wheel Selection) ---
int select_parents(Chromosome population[POOL_SIZE], Chromosome **parent1, Chromosome **parent2) {
    *parent1 = roulette_selection(population);
    if (*parent1 == nullptr) {
        return 1;
    }
    *parent2 = *parent1;
    while (*parent1 == *parent2) {
        *parent2 = roulette_selection(population);
    }
    return 0;
}

int top_chromosome_index(Chromosome population[POOL_SIZE]) {
    double lowest_loss = population[0].loss;
    int index = 0;
    for (int i = 1; i < POOL_SIZE; i++) {
        if (population[i].loss < lowest_loss) {
            lowest_loss = population[i].loss;
            index = i;
        }
    }
    return index;
}

// --- Genetic Algorithm Function ---
Chromosome genetic_algorithm(const LossFunction lossFunc, const MutationFunction mutationFunc, const CrossoverFunction crossoverFunc) {
    // Initialize population and seed random number generator
    rand_state = (uint32_t)time(nullptr); // For random seed
    Chromosome *population = initialize_population(lossFunc);
    Chromosome *previous_population = malloc(sizeof(Chromosome) * POOL_SIZE);

    double recent_best_losses[HYPERPARAMETER_HISTORY];
    double recent_best_losses_diffs[HYPERPARAMETER_HISTORY - 1];
    memset(recent_best_losses, INFINITY, sizeof(recent_best_losses));
    int recent_best_losses_index = 0;

    for (int gen = 0; gen < GENERATIONS; gen++) {
        Chromosome *new_population = malloc(sizeof(Chromosome) * POOL_SIZE);

        const int top_index = top_chromosome_index(population);
        recent_best_losses[recent_best_losses_index] = population[top_index].loss;
        for (int i = 0; i < HYPERPARAMETER_HISTORY - 1; i++) {
            const int j = (recent_best_losses_index + i + 1) % HYPERPARAMETER_HISTORY;
            recent_best_losses_diffs[i] = recent_best_losses[j] - recent_best_losses[(j + 1) % HYPERPARAMETER_HISTORY];
        }
        recent_best_losses_index = (recent_best_losses_index + 1) % HYPERPARAMETER_HISTORY;

        double sum_best_losses_diffs = 0;
        for (int i = 0; i < 9; i++) {
            sum_best_losses_diffs += recent_best_losses_diffs[i];
        }
        const double sum_average_best_losses_diffs = sum_best_losses_diffs / 9;

        if (sum_average_best_losses_diffs < hyperparameter_threshold) {
            mutation_rate *= MUTATION_RATE_CHANGE;
            crossover_rate += (1 - crossover_rate) * CROSSOVER_RATE_CHANGE;
            crossover_degree *= CROSSOVER_DEGREE_CHANGE;
            hyperparameter_threshold *= hyperparameter_threshold_change;
        }

        printf("Generation %d: Best Loss = %f, mutation rate: %f, crossover rate: %f, crossover degree: %f, average loss decrease: %f, threshold: %f\n", gen, population[top_index].loss, mutation_rate, crossover_rate, crossover_degree, sum_average_best_losses_diffs, hyperparameter_threshold);

        new_population[0] = population[top_index];
        int too_similar = 0;
        // Create new population array
        for (int i = 1; i < POOL_SIZE; i++) {
            Chromosome *parent1, *parent2;
            too_similar = select_parents(population, &parent1, &parent2);
            if (too_similar) {
                break;
            }

            Chromosome child;
            crossoverFunc(parent1, parent2, &child);
            mutationFunc(&child);

            new_population[i] = child;
        }
        if (too_similar) {
            fprintf(stderr, "Error: No longer selecting fittest chromosome because the pool is too similar. Trying to increase exploration.\n");
            mutation_rate = original_mutation_rate * pow(MUTATION_RATE_CHANGE, 1 + reset_counter * reset_multiplier);
            crossover_rate = original_crossover_rate * pow(CROSSOVER_RATE_CHANGE, 1 + reset_counter * reset_multiplier);
            crossover_degree = original_crossover_degree * pow(CROSSOVER_DEGREE_CHANGE, 1 + reset_counter * reset_multiplier);
            hyperparameter_threshold = original_hyperparameter_threshold * pow(hyperparameter_threshold_change, 1 + reset_counter * reset_multiplier);
            hyperparameter_threshold_change = 0.4;
            reset_counter++;
            free(new_population);
            memcpy(population, previous_population, sizeof(Chromosome) * POOL_SIZE);
            gen--;
            continue;
        }
#pragma omp parallel for
        for (int i = 1; i < POOL_SIZE; i++) {
            new_population[i].loss = lossFunc(&new_population[i]);
        }

        // Replace old population with the new one
        memcpy(previous_population, population, sizeof(Chromosome) * POOL_SIZE);
        memcpy(population, new_population, sizeof(Chromosome) * POOL_SIZE);

        // Save every 100 generations
        if (gen % 100 == 0) {
            PgmImage *dithered = chromosomeToPgm(&population[top_index]);
            char *filename = malloc(sizeof(char) * 100);
            sprintf(filename, "../output/dithered_ssim_adapt_glob_%i.pgm", gen);
            WritePgm(dithered, filename);

            free(dithered->data_);
            free(dithered);
            free(filename);
        }

        free(new_population);
        
    }

    // At the end, you can print the best chromosome found.
    const int best_index = top_chromosome_index(population);

    printf("Best solution after %d generations (loss %f):\n", GENERATIONS, population[best_index].loss);
    for (int i = 0; i < CHROMOSOME_SIZE; i++) {
        printf("%d ", population[best_index].genes[i]);
    }
    printf("\n");
    const Chromosome best_chromosome = population[best_index];
    free(population);
    free(previous_population);
    return best_chromosome;
}

// --- Main Function ---
// Here we pass our custom functions to the genetic algorithm.
int main(void) {
    target = ReadPgm("../images/AnyConv.com__groepsfoto_resized_smaller_scaled.pgm");

    // PbmImage *dithered_ = PgmToPbm(target, MiddleThreshold);
    //
    // PgmImage *dithered = PbmToPgm(dithered_);
    //
    // double loss = mse(target, dithered);
    //
    // printf("%f\n", loss);

    if (target->width_ * target->height_ != CHROMOSOME_SIZE) {
        fprintf(stderr, "Error: image must be of size %i\n", CHROMOSOME_SIZE);
    }
    const Chromosome result = genetic_algorithm(ssim_adaptive_and_global_loss_function, mutation_function, crossover_function);

    PgmImage *dithered = chromosomeToPgm(&result);
    WritePgm(dithered, "../output/dithered_ssim_adapt_glob.pgm");

    free(target->data_);
    free(target);
    // free(dithered_);
    free(dithered->data_);
    free(dithered);
    return 0;
}
