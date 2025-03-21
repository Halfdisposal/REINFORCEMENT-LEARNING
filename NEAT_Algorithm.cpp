#include <armadillo>
#include <SFML/Graphics.hpp>
#include <iostream>
#include "MoonLander.cpp"
#include <vector>
#include <random>
#include <algorithm>




class NEATNet{
    public:
    int input_size, hidden_size, output_size;
    arma::mat w1, b1, w2, b2;

    NEATNet(int input_size, int hidden_size, int output_size) 
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        w1 = arma::randu<arma::mat>(input_size, hidden_size);
        b1 = arma::randu<arma::mat>(1, hidden_size);
        w2 = arma::randu<arma::mat>(hidden_size, output_size);
        b2 = arma::randu<arma::mat>(1, output_size);
    }

    arma::mat ReLU(arma::mat X){
        arma::mat zeros = arma::zeros<arma::mat>(X.n_rows, X.n_cols);
        return arma::max(zeros, X);
    }

    arma::mat softmax(arma::mat X){
        arma::mat exps = arma::exp<arma::mat>(X);
        double sum_exps = arma::sum(arma::sum(X));
        return exps/sum_exps;
    }


    arma::mat forward(arma::mat X) {
        arma::mat hidden1 = X * w1 + arma::repmat(b1, X.n_rows, 1);
        arma::mat activation1 = this->ReLU(hidden1);
        arma::mat hidden2 = activation1 * w2 + arma::repmat(b2, activation1.n_rows, 1);
        arma::mat activation2 = softmax(hidden2);
        return activation2; 
    }

    std::tuple<arma::mat, arma::mat, arma::mat, arma::mat> get_params(){
        return std::make_tuple(this->w1, this->b1, this->w2, this->b2);
    }

    void set_params(arma::mat w1_n, arma::mat b1_n, arma::mat w2_n, arma::mat b2_n){
        this->w1 = w1_n;
        this->b1 = b1_n;
        this->w2 = w2_n;
        this->b2 = b2_n;
    }

    NEATNet copy(){
        NEATNet new_net = NEATNet(this->input_size, this->hidden_size, this->output_size);
        new_net.set_params(this->w1, this->b1, this->w2, this->b2);
        return new_net;
    }

    arma::mat mutate_array(const arma::mat& arr, double mutation_rate = 0.1, double mutation_strength = 0.5) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::normal_distribution<> mutation_dis(0.0, mutation_strength);
    

        arma::mat mutated_arr = arr;
    

        for (arma::uword i = 0; i < arr.n_rows; ++i) {
            for (arma::uword j = 0; j < arr.n_cols; ++j) {
                if (dis(gen) < mutation_rate) {
                    mutated_arr(i, j) += mutation_dis(gen) ;
                }
            }
        }
    
        return mutated_arr;
    }

    void mutate(float mutation_rate = 0.1, float mutation_strength = 0.5){
        this->w1 = this->mutate_array(this->w1, mutation_rate, mutation_strength);
        this->b1 = this->mutate_array(this->b1, mutation_rate, mutation_strength);
        this->w2 = this->mutate_array(this->w2, mutation_rate, mutation_strength);
        this->b2 = this->mutate_array(this->b2, mutation_rate, mutation_strength);
    }
};


NEATNet crossover(NEATNet parent1, NEATNet parent2){
    NEATNet child = parent1.copy();
    child.w1 = (parent1.w1 + parent2.w1) / 2;
    child.b1 = (parent1.b1 + parent2.b1) / 2;
    child.w2 = (parent1.w2 + parent2.w2) / 2;
    child.b2 = (parent1.b2 + parent2.b2) / 2;
    return child;

}


double evaluate(NEATNet net, MoonLanderEnv env, int episodes = 3){
    double total_reward = 0;

    for(int i = 0; i < episodes; i++){
        arma::mat state = static_cast<arma::mat>(env.reset());
        bool done = false;
        int steps = 1000;
        int step = 0;
        while (!done & step < steps){
            step += 1;
            arma::vec output = static_cast<arma::vec> (net.forward(state.t()).t());
            int action = output.index_max();
            arma::vec next_state;
            double reward;
            std::tie(next_state, reward, done) = env.step(action);
            total_reward += reward;
            state = static_cast<arma::mat>(next_state);
        }
    }
    return total_reward/(double)episodes;
}

std::vector<size_t> getEliteIndices(const std::vector<double>& fitnesses, size_t elite_num) {

    std::vector<size_t> indices(fitnesses.size());
    std::iota(indices.begin(), indices.end(), 0); 


    std::sort(indices.begin(), indices.end(),
              [&fitnesses](size_t i1, size_t i2) {
                  return fitnesses[i1] > fitnesses[i2];
              });


    if (elite_num > indices.size()) {
        elite_num = indices.size();
    }
    std::vector<size_t> elite_indices(indices.begin(), indices.begin() + elite_num);

    return elite_indices;
}




NEATNet run(MoonLanderEnv env, int generations = 10, int population_size = 50, float elite_fraction = 0.5){
    int input_size = env.input_size;
    int output_size = env.output_size;
    int hidden_size = 10;

    std::vector<NEATNet> population;

    for (int i = 0; i < population_size; i++){
        population.push_back(NEATNet(input_size, hidden_size, output_size));
    }

    for(int gen = 0; gen < generations; gen++){
        std::vector<double> fitnesses;
        for(NEATNet net: population){
            double fitness = evaluate(net, env);
            fitnesses.push_back(fitness);
        }    
        
        double maxFitness = 0.0;
        int index = 0;
        if (!fitnesses.empty()) {

            auto maxElementIter = std::max_element(fitnesses.begin(), fitnesses.end());
            maxFitness = *maxElementIter;
            auto index = std::distance(fitnesses.begin(), maxElementIter);
        } else {
            std::cout << "The vector is empty." << std::endl;
        }

        double fitness_sum = 0.0;

        for(double fitness : fitnesses){
            fitness_sum += fitness;
        }
        double avg_fitness = avg_fitness/(fitnesses.size());

        std::cout<<"Generation: "<<gen<<" Best Fitness: "<<maxFitness<<" Avg Fitness: "<<avg_fitness<<std::endl;

        int elite_num = std::max(1, (int)(elite_fraction * population_size));
        auto elite_indices = getEliteIndices(fitnesses, elite_num);
        std::vector<NEATNet> elites;
        for(int i : elite_indices){
            elites.push_back(population[i]);
        }

        std::vector<NEATNet> new_population;


        for (NEATNet elite : elites){
            new_population.push_back(elite);
        }

        //std::cout<<"New Population"<<std::endl;
        int new_population_size = new_population.size();
        while(new_population_size < population_size){
            new_population_size = new_population.size();
            //std::cout<<new_population_size<<std::endl;
            size_t num_parents = 2;


            std::random_device rd;  
            std::mt19937 gen(rd()); 
            std::uniform_int_distribution<> dis(0, elites.size() - 1);
        

            std::vector<NEATNet> parents;
            parents.reserve(num_parents);
        

            for (size_t i = 0; i < num_parents; ++i) {
                int random_index = dis(gen);
                parents.push_back(elites[random_index]);
            }

            NEATNet child = crossover(parents[0], parents[1]);
            child.mutate();
            new_population.push_back(child);
        }
        population = new_population;
    }

    int best_index = 0;
    double newmaxFitness = 0.0;
    std::vector<double> new_fitnesses;
    for(NEATNet net: population){
        double fitness = evaluate(net, env);
        new_fitnesses.push_back(fitness);
    }
    if (!new_fitnesses.empty()) {

        auto maxElementIter = std::max_element(new_fitnesses.begin(), new_fitnesses.end());
        newmaxFitness = *maxElementIter;
        auto best_index = std::distance(new_fitnesses.begin(), maxElementIter);
    } else {
        std::cout << "The New Fitness is empty." << std::endl;
    }
    NEATNet best_net = population[best_index];
    return best_net;
}







int main(){
    MoonLanderEnv env;
    NEATNet best_net = run(env, 100);
}


