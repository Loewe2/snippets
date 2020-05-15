#include <cstdio>
#include <string>
#include <cstring>
#include <iostream>
#include <cassert>
#include <exception>

int count(const std::string& str, const char* substr) {
        int occurrence = 0;
        std::size_t pos = 0;
        while((pos = str.find(substr, pos)) != std::string::npos) {
            ++occurrence;
            pos += std::strlen(substr);
        }
        return occurrence;
}
int count(const std::string& str, const std::string& substr) { return count(str, substr.c_str()); }

template <typename T, typename... Types>
constexpr bool is_type_v = (std::is_same_v<std::decay_t<T>, std::decay_t<Types>> || ...);

template <typename... Args>
void print(std::string msg, Args&&... args) {
    int occurrence = count(msg, "{}");
    assert((occurrence == sizeof...(Args)));
    std::size_t pos = 0;
    ([&](auto&& arg) {
        pos = msg.find("{}");

        if constexpr (is_type_v<decltype(arg), char>) {
            msg.replace(pos, 2, "%c");
        } else if constexpr (is_type_v<decltype(arg), short>) {
            msg.replace(pos, 2, "%hi");
        } else if constexpr (is_type_v<decltype(arg), int>) {
            msg.replace(pos, 2, "%i");
        } else if constexpr (is_type_v<decltype(arg), long>) {
            msg.replace(pos, 2, "%li");
        } else if constexpr (is_type_v<decltype(arg), long long>) {
            msg.replace(pos, 2, "%lli");
        } else if constexpr (is_type_v<decltype(arg), unsigned short>) {
            msg.replace(pos, 2, "%hu");
        } else if constexpr (is_type_v<decltype(arg), unsigned int>) {
            msg.replace(pos, 2, "%u");
        } else if constexpr (is_type_v<decltype(arg), unsigned long>) {
            msg.replace(pos, 2, "%lu");
        } else if constexpr (is_type_v<decltype(arg), float, double>) {
            msg.replace(pos, 2, "%g");
        } else if constexpr (is_type_v<decltype(arg), long double>) {
            msg.replace(pos, 2, "%Lg");
        } else if constexpr (is_type_v<decltype(arg), const char*, char*>) {
            msg.replace(pos, 2, "%s");
        } else if constexpr (std::is_pointer_v<std::decay_t<decltype(arg)>>) {
            msg.replace(pos, 2, "%p");
        } 
        else {
            using namespace std::string_literals;
            throw std::invalid_argument("Invalid argument passed! "s + typeid(arg).name());
        }

    }(args), ...);

    std::cout << msg << std::endl;
    std::printf(msg.c_str(), std::forward<Args>(args)...);
}


//int main() {
//    char c = 'a';
//    short s = 2;
//    int i = 3;
//    long l = 4;
//    long long ll = 5;
//    unsigned short us = 6;
//    unsigned int ui = 7;
//    unsigned long ul = 8;
//    std::size_t st = 9;
//    float f = 2.5;
//    double d = 3.5;
//    long double ld = 4.5;
//    std::string str = "std::string";
//    std::string_view strv = "std::string_view";
//    const char* cp = "const char*";
//    char arr[] = "char[]";
//    int* ptr = &i;
//
//    print("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}", 
//        c, s, i, l, ll, us, ui, ul, st, f, d, ld, str.c_str(), strv.data(), cp, arr, ptr);
//
//   return 0;
//}
