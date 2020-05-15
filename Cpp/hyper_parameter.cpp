#include <ostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <string>
#include <type_traits>
#include <charconv>

#include <iostream>


template <typename T>
T convert_to(const std::string& str) {
    if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
        try {
            if constexpr (std::is_same_v<std::decay_t<T>, float>) {
                return std::stof(str);
            } else if constexpr (std::is_same_v<std::decay_t<T>, double>) {
                return std::stod(str);
            } else {
                return std::stold(str);
            }
        } catch (const std::exception& e) {
            throw std::invalid_argument("Can't parse string!: " + str);
        }
    } else {
        T val;
        auto [p, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
        if (ec == std::errc()) {
            return val;
        } else {
            throw std::invalid_argument("Can't parse string!: " + str);
        }
    }
}


template <typename real_t = float, typename size_t = uint32_t, typename hash_value_t = uint32_t>
struct options {
    using real_type = real_t;
    using size_type = size_t;
    using hash_value_type = hash_value_t;


    class factory {
        template <typename T, typename U, typename V>
        friend class options;
    public:
        factory() = default;
        factory(const std::string& file) {
            if (!std::filesystem::exists(file)) {
                throw std::invalid_argument("File doesn't exist!: " + file);
            }
            
            std::ifstream in(file);
            std::string key, value;
            
            while(in >> key >> value) {
                if (key == "k") {
                    k_ = convert_to<size_type>(value);
                } else if (key == "num_hash_tables") {
                    num_hash_tables_ = convert_to<size_type>(value);
                } else if (key == "hash_table_size") {
                    hash_table_size_ = convert_to<hash_value_type>(value);
                } else if (key == "num_hash_functions") {
                    num_hash_functions_ = convert_to<size_type>(value);
                } else if (key == "w") {
                    w_ = convert_to<real_type>(value);
                } else {
                    throw std::invalid_argument("Invalid options file!: " + key);
                }
            }
        }

        factory& set_k(const size_type k) {
            k_ = k;
            return *this;
        }
        factory& set_num_hash_tables(const size_type num_hash_tables) {
            num_hash_tables_ = num_hash_tables;
            return *this;
        }
        factory& set_hash_table_size(const hash_value_type hash_table_size) {
            hash_table_size_ = hash_table_size;
            return *this;
        }
        factory& set_num_hash_functions(const size_type num_hash_functions) {
            num_hash_functions_ = num_hash_functions;
            return *this;
        }
        factory& set_w(const real_type w) {
            w_ = w;
            return *this;
        }
    private:
        size_type k_ = 6;
        size_type num_hash_tables_ = 2;
        hash_value_type hash_table_size_ = 105613;
        size_type num_hash_functions_ = 4;
        real_type w_ = 1.0;
    };


    options(options::factory fact = options<real_t, size_t, hash_value_t>::factory()) 
        : k(fact.k_), num_hash_tables(fact.num_hash_tables_), hash_table_size(fact.hash_table_size_),
          num_hash_functions(fact.num_hash_functions_), w(fact.w_) { }


    const size_type k;
    const size_type num_hash_tables;
    const hash_value_type hash_table_size;
    const size_type num_hash_functions;
    const real_type w;


    friend std::ostream& operator<<(std::ostream& out, const options& opt) {
        out << "k " << opt.k << '\n';
        out << "num_hash_tables " << opt.num_hash_tables << '\n';
        out << "hash_table_size  " << opt.hash_table_size << '\n';
        out << "num_hash_functions " << opt.num_hash_functions << '\n';
        out << "w " << opt.w << '\n';

        return out;
    }
};

int main() {

    options opt = options<>::factory().set_k(8).set_num_hash_tables(20).set_num_hash_functions(32).set_w(1.5);
    std::cout << opt << std::endl;

    options opt2;
    std::cout << opt2 << std::endl;
    
    try {
        options opt3 = options<>::factory("default_options.txt");
        std::cout << opt3 << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}

