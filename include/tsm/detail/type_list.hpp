#pragma once

#include <cstddef>
#include <type_traits>

namespace tsm::detail {

template <typename... Ts>
struct TypeList {
    static constexpr std::size_t size = sizeof...(Ts);
};

template <typename TL>
struct SizeOf;

template <typename... Ts>
struct SizeOf<TypeList<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};

template <typename TL>
inline constexpr std::size_t size_of_v = SizeOf<TL>::value;

template <typename TL, typename T>
struct Contains;

template <typename T>
struct Contains<TypeList<>, T> : std::false_type {};

template <typename T, typename... Tail>
struct Contains<TypeList<T, Tail...>, T> : std::true_type {};

template <typename Head, typename... Tail, typename T>
struct Contains<TypeList<Head, Tail...>, T> : Contains<TypeList<Tail...>, T> {};

template <typename TL, typename T>
inline constexpr bool contains_v = Contains<TL, T>::value;

template <typename TL, typename T>
struct IndexOf;

template <typename T>
struct IndexOf<TypeList<>, T> {
    static_assert(sizeof(T) == 0, "Type not found in TypeList");
};

template <typename Head, typename... Tail>
struct IndexOf<TypeList<Head, Head, Tail...>, Head>
    : std::integral_constant<std::size_t, 0> {};

template <typename T, typename... Tail>
struct IndexOf<TypeList<T, Tail...>, T>
    : std::integral_constant<std::size_t, 0> {};

template <typename Head, typename... Tail, typename T>
struct IndexOf<TypeList<Head, Tail...>, T>
    : std::integral_constant<std::size_t, 1 + IndexOf<TypeList<Tail...>, T>::value> {};

template <typename TL, typename T>
inline constexpr std::size_t index_of_v = IndexOf<TL, T>::value;

template <typename TL, std::size_t I>
struct TypeAt;

template <typename Head, typename... Tail>
struct TypeAt<TypeList<Head, Tail...>, 0> {
    using type = Head;
};

template <typename Head, typename... Tail, std::size_t I>
struct TypeAt<TypeList<Head, Tail...>, I> {
    using type = typename TypeAt<TypeList<Tail...>, I - 1>::type;
};

template <typename TL, std::size_t I>
using type_at_t = typename TypeAt<TL, I>::type;

template <typename TL1, typename TL2>
struct Concat;

template <typename... Ts, typename... Us>
struct Concat<TypeList<Ts...>, TypeList<Us...>> {
    using type = TypeList<Ts..., Us...>;
};

template <typename TL1, typename TL2>
using concat_t = typename Concat<TL1, TL2>::type;

template <typename TL, std::size_t N>
struct Head;

template <>
struct Head<TypeList<>, 0> {
    using type = TypeList<>;
};

template <typename H, typename... Ts>
struct Head<TypeList<H, Ts...>, 0> {
    using type = TypeList<>;
};

template <typename H, typename... Ts, std::size_t N>
    requires (N > 0)
struct Head<TypeList<H, Ts...>, N> {
    using type = concat_t<TypeList<H>, typename Head<TypeList<Ts...>, N - 1>::type>;
};

template <typename TL, std::size_t N>
using head_t = typename Head<TL, N>::type;

template <typename TL, std::size_t N>
struct Tail;

template <>
struct Tail<TypeList<>, 0> {
    using type = TypeList<>;
};

template <typename H, typename... Ts>
struct Tail<TypeList<H, Ts...>, 0> {
    using type = TypeList<H, Ts...>;
};

template <typename H, typename... Ts, std::size_t N>
    requires (N > 0)
struct Tail<TypeList<H, Ts...>, N> {
    using type = typename Tail<TypeList<Ts...>, N - 1>::type;
};

template <typename TL, std::size_t N>
using tail_t = typename Tail<TL, N>::type;

template <typename TL, std::size_t From, std::size_t Count>
using sub_list_t = head_t<tail_t<TL, From>, Count>;

template <typename TL1, typename TL2>
struct AreIdentical : std::false_type {};

template <typename... Ts>
struct AreIdentical<TypeList<Ts...>, TypeList<Ts...>> : std::true_type {};

template <typename TL1, typename TL2>
inline constexpr bool are_identical_v = AreIdentical<TL1, TL2>::value;

template <typename TL1, typename TL2>
struct IsSubsetOf;

template <typename TL2>
struct IsSubsetOf<TypeList<>, TL2> : std::true_type {};

template <typename Head, typename... Tail, typename TL2>
struct IsSubsetOf<TypeList<Head, Tail...>, TL2>
    : std::conjunction<Contains<TL2, Head>, IsSubsetOf<TypeList<Tail...>, TL2>> {};

template <typename TL1, typename TL2>
inline constexpr bool is_subset_of_v = IsSubsetOf<TL1, TL2>::value;

} // namespace tsm::detail
