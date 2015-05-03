---
layout: page
title: Learn You a Haskell for Great Good
author: Miran Lipovaca
category: Programming
tags: Haskell
ISBN: 1593272839
---

- **Chapter 1 - Starting Out**
    - *1.1 - Ready, Set, Go!*
        - Invalid: `5 * -3`, valid `5 * (-3)`
        - Function application has the highest precedence
            `succ 9 + max 5 4 + 1` is equivalent to `(succ 9) + (max 5 4) + 1`
        - If a function takes two parameters, we can also call it as an infix function by surrounding it with backticks
            `div 92 10` is equivalent to ``92 `div` 10``
    - *1.2 - Baby's First Functions*
        - If must have an else

            ```haskell
            doubleSmallNumber x = if x > 100 then x else x * 2
            ```
        - Function names can't start with uppercase, because uppercase means a type
        - Function can have arity zero

            ```haskell
            conanO'Brien = "It's a-me, Conan O'Brien!"
            ```
    - *1.3 - An Intro To Lists*
        - List must contain elements of same type
        - 'let' keyword is used to define names in GHCI

            ```haskell
            let lostNumbers = [4,8,15,16,23,42]
            lostNumbers -- [4,8,15,16,23,42]
            ```
        - Single quote means character, double quotes means string
        - String is just a list of characters
        - Lists can be contenated by `++` operator, complexity is linear

            ```haskell
            [1,2,3,4] ++ [9,10,11,12] -- [1,2,3,4,9,10,11,12]
            "hello" ++ " " ++ "world" -- "hello world"
            ['w','o'] ++ ['o','t'] -- "woot"
            ```
        - `:` operator (or *cons* operator) adds an element at the beginning of a lost (`O(1)`)

            ```haskell
            'A':" SMALL CAT" -- "A SMALL CAT"
            5:[1,2,3,4,5] -- [5,1,2,3,4,5]
            ```
        - `[1, 2, 3]` is equivalent to `1:2:3:[]`
        - `!!` operator is used to access list elements

            ```haskell
            [9.4,33.2,96.2,11.2,23.25] !! 1` -- 33.2
            ```
        - Lists are compared in lexicographical order
        - Some list operations

            ```haskell
            head [5,4,3,2,1] -- 5
            tail [5,4,3,2,1] -- [4,3,2,1]
            last [5,4,3,2,1] -- 1
            init [5,4,3,2,1] -- [5,4,3,2]
            length [5,4,3,2,1] -- 5
            null [1,2,3] -- False
            null [] -- True
            reverse [5,4,3,2,1] -- [1,2,3,4,5]
            take 3 [5,4,3,2,1] -- [5,4,3]
            take 5 [1,2] -- [1,2]
            drop 3 [8,4,2,1,5,6] -- [1,5,6]
            drop 100 [1,2,3,4] -- []
            minimum [8,4,2,1,5,6] -- 1
            maximum [1,9,2,3,4] -- 9
            sum [5,2,1,6,3,2,5,7] -- 31
            product [6,2,1,2] -- 24
            elem 5 [1,3,5,7] -- True
            elem 4 [1,3,5,7] -- False
            ```
    - *1.4 - Texas Ranges*
        - Range example

            ```haskell
            ['K'..'Z'] -- "KLMNOPQRSTUVWXYZ"
            [3,6..20] -- [3,6,9,12,15,18]
            [5,4..1] -- [5,4,3,2,1]
            [13,26..] -- [13,26,39,52,...]
            ```
        - New functions

            ```haskell
            cycle [1,2,3] -- [1,2,3,1,2,3,...]
            repeat 5 -- [5,5,5,...]
            replicate 3 10 -- [10,10,10]
            ```
    - *1.5 - I'm A List Comprehension*
        - List comprehension examples

            ```haskell
            [x*2 | x <- [1..10]] -- [2,4,6,8,10,12,14,16,18,20]
            [x | x <- [50..100], x `mod` 7 == 3] -- [52,59,66,73,80,87,94]
            [x*y | x <- [2,5,10], y <- [8,10,11], x*y > 50] -- [55,80,100,110]
            length' xs = sum [1 | _ <- xs]
            take 5 [(i,j) | i <- [1..], j <- [1..i-1], gcd i j == 1] -- [(2,1),(3,1),(3,2),(4,1),(4,3)]
            take 5 [ (i,j) | i <- [1..], let k = i*i, j <- [1..k]] -- [(1,1),(2,1),(2,2),(2,3),(2,4)]
            ```
    - *1.6 - Tuples*
        - Can have different types of elements
        - Has its own type depending on its size and type of its elements
        - New functions

            ```haskell
            fst ("Wow", False) -- "Wow", works only on pair
            snd (8,11) -- 11, works only on pair
            zip [5,3,2,6,2,7,2,5,4,6,6] ["im","a","turtle"] -- [(5,"im"),(3,"a"),(2,"turtle")]
            ```
- **Chapter 2 - Believe The Type**
    - *2.1 - Explicit Type Declaration*
    - *2.2 - Common Haskell Types*
        - Some basic types: Bool, Char, Int, Float, Double, Integer (arbitrary precision integer)
        - Declaring functions with type

            ```haskell
            addThree :: Int -> Int -> Int -> Int
            addThree x y z = x + y + z
            factorial :: Integer -> Integer
            factorial n = product [1..n]
            factorial 50 -- 30414093201713378043612608166064768844377641568960512000000000000
            ```
    - *2.3 - Type Variables*
    - *2.4 - Typeclasses 101*
        - If a function is comprised only of special char- acters, it's considered an infix function by default. If we want to examine its type, pass it to another function or call it as a prefix function, we have to surround it in parentheses.

            ```haskell
            (==) 8 8 -- True
            (+) 7 8 -- 15
            ```
        - `Eq` typeclass provides an interface for types that can be tested for equality
        - `Ord` typeclass provides an interface for types that can be ordered
        - The `compare` function takes two `Ord` members of the same type and returns an ordering. Ordering is a type that can be `GT`, `LT` or `EQ`, meaning greater than, lesser than and equal, respectively.

            ```haskell
            "Abrakadabra" `compare` "Zebra" -- LT
            5 `compare` 3 -- GT
            ```
        - `show` transforms members of `Show` typeclass to `String`
        - `read` takes a string and returns a type of 'Read' typeclass
        - `X::Type` can be used to transform a typeclass `X` to `Type` whre `Type` is of typeclass `X`

            ```haskell
            show 5.334 -- "5.334"
            read "[1,2,3,4]" :: [Int] -- [1,2,3,4]
            read "(3, 'a')" :: (Float, Char) -- (3.0, 'a')
            minBound :: Int -- -2147483648
            maxBound :: Char -- '\1114111'
            maxBound :: (Bool, Int, Char) -- (True,2147483647,'\1114111')
            ```
        - Some other typeclasses: `Integral`, `Floating
- **Chapter 3 - Syntax In Functions**
    - *3.1 - Pattern matching*
        - Examples

            ```haskell
            addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
            addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

            first :: (a, b, c) -> a
            first (x, _, _) = x

            second :: (a, b, c) -> b
            second (_, y, _) = y

            third :: (a, b, c) -> c
            third (_, _, z) = z


            let xs = [(1,3), (4,3), (2,4), (5,3), (5,6), (3,1)]
            [a+b | (a,b) <- xs] -- [4,7,6,8,11,4]

            head' :: [a] -> a
            head' [] = error "Can't call head on an empty list, dummy!"
            head' (x:_) = x
            ```
        - *as-patterns*: `@` can be used to refer patterns to use later

            ```haskell
            capital :: String -> String
            capital "" = "Empty string, whoops!"
            capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]
            capital "Dracula" -- "The first letter of Dracula is D"
            ```
    - *3.2 - Guards, Guards!*
        - A *guard* is indicated by a pipe character (|), followed by a Boolean expression, followed by the function body that will be used if that expression evaluates to `True`. If the expression evaluates to `False`, the function drops through to the next guard. **Guards must be indented.**

            ```haskell
            max' :: (Ord a) => a -> a -> a
            max' a b
                | a <= b    = b
                | otherwise = a
            ```
    - *3.3 - Where?!*
        - Examples

            ```haskell
            bmiTell :: Double -> Double -> String
            bmiTell weight height
                | bmi <= skinny = "You're underweight, you emo, you!"
                | bmi <= normal = "You're supposedly normal. Pffft, I bet you're ugly!"
                | bmi <= fat    = "You're fat! Lose some weight, fatty!"
                | otherwise     = "You're a whale, congratulations!"
                where bmi = weight / height ^ 2 -- all variable names must be aligned in a single column
                      (skinny, normal, fat) = (18.5, 25.0, 30.0)

            initials :: String -> String -> String
            initials firstname lastname = [f] ++ ". " ++ [l] ++ "."
                where (f:_) = firstname
                      (l:_) = lastname

            calcBmis :: [(Double, Double)] -> [Double]
            calcBmis xs = [bmi w h | (w, h) <- xs]
                where bmi weight height = weight / height ^ 2
            ```
    - *3.4 - let It Be*
        - `let <bindings> in <expression>`

            ```haskell
            cylinder :: Double -> Double -> Double
            cylinder r h =
                let sideArea = 2 * pi * r * h
                    topArea = pi * r ^ 2
                in sideArea + 2 * topArea
            ```

        - `let` is an expression, so it has a value
        - `let` can be separated with semicolons in the same line

            ```haskell
            4 * (let a = 9 in a + 1) + 2 -- 42
            let square x = x * x in (square 5, square 3, square 2) -- (25, 9, 4)

            (let (a, b, c) = (1, 2, 3) in a*b*c, let foo="Hey "; bar = "there!" in foo ++ bar) -- (6,"Hey there!")
            ```
        - 'let' in list comprehension


            ```haskell
            calcBmis :: [(Double, Double)] -> [Double]
            calcBmis xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2, bmi > 25.0]
            ```
            The `(w, h) <- xs` is called *generator*. `bmi` can't be refered in the generator, because that is defined prior to the `let` binding.
        - `case` expression

            ```
            case <expression> of pattern -> result
                                 pattern -> result
                                 pattern -> result
                                 ...
            ```
            Example:

            ```haskell
            head' :: [a] -> a
            head' xs = case xs of [] -> error "No head for empty lists!"
                                  (x:_) -> x

            describeList :: [a] -> String
            describeList ls = "The list is " ++ case ls of [] -> "empty."
                                                           [x] -> "a singleton list."
                                                           xs -> "a longer list."
            ```
- **Chapter 4 - Hello Recursion!**
    - *4.1 - Maximum Awesome*
    - *4.2 - A Few More Recursive Functions*
    - *4.3 - Quick, Sort!*

        ```haskell
        quicksort :: (Ord a) => [a] -> [a]
        quicksort [] = []
        quicksort (x:xs) = quicksort left ++ [x] ++ quicksort right
            where left = [a | a <- xs, a <= x]
                  right = [a | a <- xs, a > x]
        ```
- **Chapter 5 - Higher-Order Functions**
    - *5.1 - Curried Functions*
        - Infix functions can be curried by *sections*. ``a `func` b`` is equivalent to ``(`func` b) a``
        - Be careful about negetive numbers. `(-4)` should have been ``(subtract 4)``, instead it is just `-4`
    - *5.2 - Some Higher-Orderism Is in Order*
        - Implementing flip

            ```haskell
            flip' :: (a -> b -> c) -> (b -> a -> c)
            flip' f = g
                where g x y = f y x
            -- another version
            flip' f y x = f x y -- this works because flip' f x y also defines flip' f x and flip' f
            ```
    - *5.3 - The Functional Programmer’s Toolbox*
        - `takeWhile :: (a -> Bool) -> [a] -> [a]` returns begining of a list until `f` is true

            ```haskell
            takeWhile (< 5) [1..] -- [1, 2, 3, 4]
            ```
    - *5.4 - Lambdas*
        - Examples

            ```haskell
            map (\(a,b) -> a + b) [(1,2),(3,5),(6,3),(2,6),(2,5)] -- [3,8,9,8,7]

            flip' :: (a -> b -> c) -> b -> a -> c
            flip' f = \x y -> f y x -- \x y -> (f x y)


            addThree :: Int -> Int -> Int -> Int
            addThree' = \x -> \y -> \z -> x + y + z -- (\x -> (\y -> (\z -> x + y + z)))
            ```
    - *5.5 - I Fold You So*
        - `foldl acc init [x1, x2, ..., xn] = acc (... (acc init x1) ...) xn`
        - `foldr acc init [x1, x2, ..., xn] = acc x1 (... (acc xn init) ...)`
        - `foldr` works on infinite list
        - The `scanl` and `scanr` functions are like foldl and foldr, except they report all the intermediate accumulator states in the form of a list

        ```haskell
        sum' :: (Num a) => [a] -> a
        sum' = foldl (\acc x -> acc + x) 0 -- foldl (+) 0

        map' :: (a -> b) -> [a] -> [b]
        map' f = foldr (\x acc -> f x : acc) []

        elem' :: (Eq a) => a -> [a] -> Bool
        elem' y ys = foldr (\x acc -> if x == y then True else acc) False ys

        reverse' :: [a] -> [a]
        reverse' = foldl (\acc x -> x : acc) [] -- foldl (flip(:)) []

        filter' :: (a -> Bool) -> [a] -> [a]
        filter' f = foldr (\x acc -> if f x then x : acc else acc) []

        scanl (+) 0 [3,5,2,1] -- [0,3,8,10,11]
        scanr (+) 0 [3,5,2,1] -- [11,8,3,1,0]
        ```
    - *5.6 - Function Application with $*
        - *Function Application Operator* `$` is defined as

            ```haskell
            ($) :: (a -> b) -> a -> b
            f $ x = f x
            ```
        - It has lowest precedence, hence right associative

            ```haskell
            f a b c -- ((f a) b) c
            f $ g $ h $ a -- f (g (h a))
            ```
        - It allows to treat function application like another function

            ```haskell
            ($ a) f -- f a
            map ($ 3) [(4+), (10*), (^2), sqrt] -- [7.0,30.0,9.0,1.7320508075688772]
            ```
    - *5.7 - Function Composition*
        - *Function Composition Operator* `.` is defined as

            ```haskell
            (.) :: (b -> c) -> (a -> b) -> a -> c
            f . g = \x -> f (g x)
            ```
        - It is right associative
        - It is used to write functions in *point-free* style

        ```haskell
        map (negate . sum . tail) [[1..5],[3..6],[1..7]] -- [-14,-15,-27]

        replicate 2 (product (map (*3) (zipWith max [1,2] [4,5])))
        -- can be turned into
        replicate 2 $ product $ map (*3) $ zipWith max [1,2] [4,5]
        -- which is equivalent to
        replicate 2 . product . map (*3) $ zipWith max [1,2] [4,5]

        fn x = ceiling (negate (tan (cos (max 50 x))))
        -- can be rewritten in point-free style as
        fn = ceiling . negate . tan . cos . max 50
        ```
- **Chapter 6 - Modules**
    - *6.1 - Importing Modules*

        ```haskell
        import Data.List -- imports Data.List module
        :m + Data.List Data.Map Data.Set -- imports modules in ghci
        import Data.List (nub, sort) -- imports nub and sort functions from Data.List
        import Data.List hiding (nub, sort) -- imports everything but nub and sort from Data.List
        import qualified Data.Map (filter) -- imports filter from Data.Map, but filter should be accessed as Data.Map.filter
        import qualified Data.Map as M (filter) -- imports filter from Data.Map, but filter should be accessed as M.filter
        ```
    - *6.2 - Solving Problems with Module Functions*
        - Some functions

            ```haskell
            import Data.List (group, sort, words, tails, isPrefixOf, isInfixOf, any, all, foldl', foldr')

            words "hey these are the words in this sentence" -- ["hey","these","are","the","words","in","this","sentence"]
            group [1,1,1,1,2,2,2,2,3,3,2,2,2,5,6,7] -- [[1,1,1,1],[2,2,2,2],[3,3],[2,2,2],[5],[6],[7]]
            sort "abracadabra" -- aaaaabbcdrr
            tails [1,2,3] -- [[1,2,3],[2,3],[3],[]]
            hawaii" `isPrefixOf` "hawaii joe" -- True
            "art" `isInfixOf` "party" -- True
            any (>5) [1..] -- True
            all (>5) [1..] -- False
            foldl (+) 0 (replicate 10000000 1) -- Stack Overflow
            foldl' (+) 0 (replicate 10000000 1) -- 10000000, foldl' doesn't defer computation

            import Data.Char (ord, chr, digitToInt)

            ord 'a' -- 97
            chr 97 -- 'a'
            map digitToInt (['0'..'9']++['A'..'F']++['a'..'f']) -- [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,10,11,12,13,14,15]

            ```
        - `find` function has type `(a -> Bool) -> [a] -> Maybe a`

            ```haskell
            find (<4) [1..] -- Just 4
            find (<4) [4..] -- Nothing
            ```
    - *6.3 - Mapping Keys to Values*
        - `O(n)` association-list implementation

            ```haskell
            findKey :: (Eq k) => k -> [(k, v)] -> Maybe v
            findKey key xs = foldr (\(k, v) acc -> if key == k then Just v else acc) Nothing xs
            ```
        - `Data.Map` examples

            ```haskell
            import qualified Data.Map as Map

            -- Map.fromList :: (Ord k) => [(k, v)] -> Map.Map k v
            -- takes a list of pairs and returns a map from type k to v
            squares = Map.fromList [(i, i^2) | i <- [1..5]]
            Map.lookup 4 squares -- Just 16
            Map.lookup 6 squares -- Nothing

            -- Map.insert :: (Ord k) => k -> a -> Map.Map k a -> Map.Map k a
            squares' = Map.insert 6 (6^2) squares
            Map.lookup 6 squares' -- Just 36

            -- Map.size :: Map.Map k a -> Int
            Map.size squares -- 5

            -- Map.fromListWith :: Ord k => (a -> a -> a) -> [(k, a)] -> Map.Map k a
            -- Takes a function and apply that on duplicates
            Map.fromListWith (++) [(1, [1]), (1, [2]), (2, [1])] -- fromList [(1,[2,1]),(2,[1])]
            Map.fromListWith max [(2,3),(2,5),(2,100),(3,29),(3,22),(3,11),(4,22),(4,15)] -- fromList [(2,100),(3,29),(4,22)]
    - *6.4 - Making Our Own Modules*
        - Sample module

            ```haskell
            module TestModule ( test ) where
            test x = x ^ 2 -- export
            test' x = x ^ 3 -- doesn't export
            ```
        - Submodules resides under directory of parent modules

            ```haskell
            module TestModule.SubModule ( test ) where
            test x = x ^ 2 -- export
            ```
- **Chapter 7 - Making Our Own Types And Type Classes**
    - *7.1 - Defining a New Data Type*

        ```haskell
        data Bool = False | True

        :t True -- Bool
        :t False -- Bool
        ```
    - *7.2 - Shaping Up*

        ```haskell
        data Point = Point Float Float deriving (Show)
        data Shape = Circle Point Float | Rectangle Point Point deriving (Show)

        :t Circle -- Point -> Float -> Shape
        :t Rectangle -- Point -> Point -> Shape

        area :: Shape -> Float
        area (Circle _ r) = pi * r ^ 2
        area (Rectangle (Point x1 y1) (Point x2 y2)) = (abs $ x2 - x1) * (abs $ y2 - y1)
        ```
        - To export constructor from module use `(..)`

            ```haskell
            module Point ( Point(..), ... ) where
            ...
            ```
    - *7.3 - Record Syntax*

        ```haskell
        data Point = Point {x :: Int, y :: Int} deriving (Show)

        Point 8 9 -- Point {x = 8, y = 9}
        Point {y = 8, x = 9} -- Point {x = 9, y = 8}
        x $ Point 6 7 -- 6
        ```
    - *7.4 - Type Parameters*

        ```haskell
        data Vector a = Vector a a a deriving (Show)

        vplus :: (Num a) => Vector a -> Vector a -> Vector a
        -- Note that we used Vector a instead of Vector a a a
        -- because Vector a a a is the constructor but Vector a is Type
        (Vector i j k) `vplus` (Vector l m n) = Vector (i+l) (j+m) (k+n)
        ```
    - *7.5 - Derived Instances*

        ```haskell
        data Point a = Point {x :: a, y :: a} deriving (Eq, Show, Read)

        (Point 6 7) == (Point 6 7) -- True
        show $ Point 6 7 -- "Point {x = 6, y = 7}"
        read "Point {x = 6, y = 7}" :: Point Int -- Point {x = 6, y = 7}
        read "Point {x = 6, y = 7}" == Point 8 9 -- :: Point Int part is not needed because haskell can now deduce the type from Point 8 9


        data Day = Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday deriving (Eq, Ord, Show, Read, Bounded, Enum)

        Monday < Friday -- True
        minBound :: Day -- Monday
        maxBound :: Day -- Sunday
        succ Monday -- Tuesday
        pred Saturday -- Friday
        [Thursday .. Sunday] -- [Thursday,Friday,Saturday,Sunday]
        [minBound .. maxBound] :: [Day] -- [Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday]
        ```

        - If two values were made using the same constructor, they are considered to be equal, unless they have fields. If they have fields, the fields are compared to see which is greater. (in this case, the types of the fields also must be part of the Ord type class.)

            ```haskell
            data Maybe a = Nothing | Just a deriving (Eq, Ord)

            Nothing < Just 100 -- True, because Nothing is declared before Just
            Just 100 > Just 50 -- True, because Just 100 > Just 50 is reduced to 100 > 50
            ```
    - *7.6 - Type Synonyms*
        - Type synonyms are used to give some types different names. They, too, can be parameterized.

            ```haskell
            type String = [Char]
            type AssocList k v = [(k, v)]
            ```
        - `Either` data type

            ```haskell
            data Either a b = Left a | Right b deriving (Eq, Ord, Read, Show)
            ```
    - *7.7 - Recursive data structures*
        - Defining custop list

            ```haskell
            data List a = Empty | Cons { listHead :: a, listTail :: List a} deriving (Show, Read, Eq, Ord)

            a = Cons 1 $ Cons 2 $ Cons 3 $ Empty
            listHead a -- 1
            listTail a -- Cons {listHead = 2, listTail = Cons {listHead = 3, listTail = Empty}}
            ```
        - fixity declarations

            ```haskell
            infixr 5 :-:
            -- infixr means the :-: operator is right associative
            -- 5 is called fixity, which states the precedence of the operator (0-9)
            -- for example, * has fixity 7, + has 6
            data List a = Empty | a :-: (List a) deriving (Show, Read, Eq, Ord)

            (3 :-: 4 :-: 5 :-: Empty) -- 3 :-: (4 :-: (5 :-: Empty))

            -- :-: can noe be used to match pattern
            infixr 5  .++
            (.++) :: List a -> List a -> List a
            Empty .++ ys = ys
            (x :-: xs) .++ ys = x :-: (xs .++ ys) -- :-: is used in pattern
            ```
        - Binary Search Tree

            ```haskell
            data Tree a = EmptyTree | Node a (Tree a) (Tree a) deriving (Show, Read, Eq)

            treeInsert :: (Ord a) => a -> Tree a -> Tree a
            treeInsert x EmptyTree = Node x EmptyTree EmptyTree
            treeInsert x (Node a left right)
                | x < a     = Node a (treeInsert x left) right
                | x > a     = Node a left (treeInsert x right)
                | otherwise = (Node a left right)

            treeElem :: (Ord a) => a -> Tree a -> Bool
            treeElem x EmptyTree = False
            treeElem x (Node a left right)
                | x < a     = treeElem x left
                | x > a     = treeElem x right
                | otherwise = True

            tree = foldr treeInsert EmptyTree [8,6,1,7,3]
            -- Node 3 (Node 1 EmptyTree EmptyTree) (Node 7 (Node 6 EmptyTree EmptyTree) (Node 8 EmptyTree EmptyTree))
            treeElem 6 tree -- True
            treeElem 5 tree -- False
            ```
    - *7.8 - Typeclasses 102*
        - Typeclass definition example

            ```haskell
            class Eq a where
                (==) :: a -> a -> Bool
                (/=) :: a -> a -> Bool
                x == y = not (x /= y)
                x /= y = not (x == y)
            ```
            Because `==` was defined in terms of `/=` and vice versa in the class declaration, we only had to overwrite one of them in the instance declaration. That's called the minimal complete definition for the typeclass — the minimum of functions that we have to implement so that our type can behave like the class advertises. To fulfill the minimal complete definition for Eq, we have to overwrite either one of `==` or `/=`.
        - Examples of typeclass instanciation

            ```haskell
            data TrafficLight = Red | Yellow | Green

            instance Eq TrafficLight where
                Red == Red = True
                Green == Green = True
                Yellow == Yellow = True
                _ == _ = False
                -- note that /= are defined if == are defined

            instance Show TrafficLight where
                show Red = "Red light"
                show Yellow = "Yellow light"
                show Green = "Green light"
            ```
        - Defining typeclasses which are subclasses of other typeclasses

            ```haskell
            class (Eq a) => Num a where
                ...
            ```
        - Some examples

            ```haskell
            instance Eq Maybe where
                ...
            -- invalid,  Maybe is not a concrete type
            -- but function parameters must be of concrete type
            -- so (==) :: Maybe -> Maybe -> Bool is invalid

            instance Eq (Maybe m) where
                Just x == Just y = x == y
                Nothing == Nothing = True
                _ == _ = False
            -- valid, but will fail if Type m is not a subclass of Eq

            instance (Eq m) => Eq (Maybe m) where
                Just x == Just y = x == y
                Nothing == Nothing = True
                _ == _ = False
            -- always valid
            ```
        - `:info` in GHCI

            ```haskell
            > :info Num

            class Num a where
                (+) :: a -> a -> a
                (*) :: a -> a -> a
                (-) :: a -> a -> a
                negate :: a -> a
                abs :: a -> a
                signum :: a -> a
                fromInteger :: Integer -> a
                    -- Defined in `GHC.Num'
            instance Num Integer -- Defined in `GHC.Num'
            instance Num Int -- Defined in `GHC.Num'
            instance Num Float -- Defined in `GHC.Float'
            instance Num Double -- Defined in `GHC.Float'

            > :info (+)

            class Num a where
                (+) :: a -> a -> a
                ...
                    -- Defined in `GHC.Num'
            infixl 6 +

            > :info elem

            elem :: Eq a => a -> [a] -> Bool    -- Defined in `GHC.List'
            infix 4 `elem`
            ```
    - *7.9 - A yes-no typeclass*
    - *7.10 - The Functor typeclass*

        ```haskell
        class Functor f where
            fmap :: (a -> b) -> f a -> f b
        ```
        - Example on list

            ```haskell
            instance Functor [] where
                fmap = map
            ```
            `[]` is a type constructor, for example `[] Int` produces type `[Int]`
        - Example on Maybe type

            ```haskell
            instance Functor Maybe where
                fmap f (Just x) = Just (f x)
                fmap f Nothing = Nothing
            ```
        - Example on `Tree` defined in 7.7

            ```haskell
            instance Functor Tree where
                fmap f EmptyTree = EmptyTree
                fmap f (Node x leftsub rightsub) = Node (f x) (fmap f leftsub) (fmap f rightsub)
            ```
    - *7.11 - Kinds and some type-foo*

        > Type constructors take other types as parameters to eventually produce concrete types. That kind of reminds us of functions, which take values as parameters to produce values. We've seen that type constructors can be partially applied (`Either String` is a type that takes one type and produces a concrete type, like `Either String Int`), just like functions can.

        ```haskell
        > :k Int
        Int :: *
        -- A * means that the type is a concrete type. A concrete type is a type that doesn't take any type parameters and values can only have types that are concrete types.

        > :k Maybe
        Maybe :: * -> *
        -- The Maybe type constructor takes one concrete type (like Int) and then returns a concrete type like Maybe Int

        > :k Either
        Either :: * -> * -> *
        -- This tells us that Either takes two concrete types as type parameters to produce a concrete type.

        > :k Either String
        Either String :: * -> *

        class Tofu t where
            tofu :: j a -> t a j
        -- a has kind *, hence j has kind * -> *
        -- so t has kind * -> (* -> *) -> *

        data Frank a b  = Frank {frankField :: b a} deriving (Show)
        -- An example of kind * -> (* -> *) -> *

        > :k Frank
        * -> (* -> *) -> *
        > :t Frank {frankField = Just "HAHA"}
        Frank {frankField = Just "HAHA"} :: Frank [Char] Maybe
        > :t Frank {frankField = "YES"}
        Frank {frankField = "YES"} :: Frank Char []

        -- Making Frank an instance of Tofu
        instance Tofu Frank where
            tofu x = Frank x
        -- Equivalent to tofu (j a) = Frank {frankField = j a}
        -- Which is of type t a j, where t = Frank

        data Barry t k p = Barry { yabba :: p, dabba :: t k } deriving (Show)

        > :k Barry
        Barry :: (* -> *) -> * -> * -> *

        instance Functor (Barry a b) where
            -- fmap :: (a -> b) -> Barry c d a -> Barry c d b
            fmap f (Barry {yabba = x, dabba = y}) = Barry {yabba = f x, dabba = y}

        fmap (^9) $ Barry 2 (Just 9) :: Barry Maybe Int Int
        -- Barry {yabba = 512, dabba = Just 9}
        ```
- **Chapter 8 - Input And Output**
    - *8.1 - Separating the Pure from the Impure*
    - *8.2 - Hello, World!*

        ```haskell
        -- helloworld.hs
        main = putStrLn "hello, world"

        -- $ ghc --make helloworld
        -- $ ./helloworld

        :t putStrLn
        putStrLn :: String -> IO () -- putStrLn returns an IO action that has result type ()
        ```
    - *8.3 - Gluing I/O Actions Together*
        - `getLine` is an I/O action that yields a `String`

            ```haskell
            > :t getLine
            getLine :: IO String
            ```
        - `<-` construct is used to bind IO results to a name

            ```haskell
            name <- getLine
            -- getLine returns a IO String
            -- hence name has type String

            _ <- putStrLn "Hello World!"
            -- putStrLn retuns a IO ()
            -- hence _ has type ()
            ```
        - `let` is used inside `do` block to bind pure values to names

            ```haskell
            main = do
                let hello = "Hello, World!"
                putStrLn hello
            ```
        - `return :: Monad m => a -> m a` makes an IO action that yields a result

            ```haskell
            > a <- return 4 :: IO Int
            -- equivalent to let a = 4 :: Int
            ``
            (using `return` doesn’t cause the I/O do block to end its execution)
    - *8.4 - Some Useful I/O Functions*

        ```haskell
        putStr -- String -> IO ()
        putStrLn -- String -> IO ()
        putChar -- Char -> IO ()
        print -- Show a => a -> IO ()

        -- import Control.Monad (when, forever)
        when -- Monad m => Bool -> m () -> m ()
        when (input == "SWORDFISH") $ do
            putStrLn input

        forever -- Monad m => m a -> m b
        -- takes an I/O action and returns an I/O action that just repeats the I/O action it got forever.
        forever $ do
            l <- getLine
            putStrLn l

        sequence -- Monad m => [m a] -> m [a]
        res <- sequence $ map print [1..3]
        -- prints 1..3 and res becomes [(), (), ()]

        mapM -- Monad m => (a -> m b) -> [a] -> m [b]
        mapM_ -- Monad m => (a -> m b) -> [a] -> m ()
        -- mapM f a = sequence $ map f a

        -- import Control.Monad (forM, forM_)
        forM -- Monad m => [a] -> (a -> m b) -> m [b]
        forM_ -- Monad m => [a] -> (a -> m b) -> m ()
        -- same as mapM but parameters reversed
        do
            lines <- forM [1..5] $ \a -> do
                line <- readLn :: IO Int
                return $ line * line
            mapM_ print lines
        ```
- **Chapter 9 - More Input And More Output**
    - *9.1 - Files and Streams*
        - `getContents :: IO String` reads everything from stdio (lazily) until EOF

            ```haskell
            main = do
                contents <- getContents
                putStr . unlines . map (\w -> ">> " ++ w) . words $ contents
            ```
        - `lines` split a string at '\n', `unlines` is the inverse of `lines`
        - `interact :: (String -> String) -> IO ()` takes a function as parameter and returns an IO action that will take some input, run that function on it, and then print out the function's result.

            ```haskell
            main = interact shortLinesOnly

            shortLinesOnly :: String -> String
            shortLinesOnly = unlines . filter (\line -> length line < 10) . lines
            ```
    - *9.2 - Reading and Writing Files*
        - `Handle`
            
            ```haskell
            import System.IO
            
            main = do
                handle <- openFile "sample.txt" ReadMode
                -- openFile :: FilePath -> IOMode -> IO Handle
                -- type FilePath = String
                -- data IOMode = ReadMode | WriteMode | AppendMode | ReadWriteMode

                contents <- hGetContents handle
                -- hGetContents :: Handle -> IO String
                
                putStr contents
                hClose handle
                -- hClose :: Handle -> IO ()
            ```
        - `withFile :: FilePath -> IOMode -> (Handle -> IO a) -> IO a`

            ```haskell
            import System.IO

            main = do
                withFile "sample.txt" ReadMode $ \handle -> do
                    contents <- hGetContents handle
                    putStr contents
            ```
        - `bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c` takes a resource, a function that releases that resource and another function that returns some other resource. Implementation of `withFile` with `bracket`:

            ```haskell
            import Control.Exception (bracket)

            withFile :: FilePath -> IOMode -> (Handle -> IO a) -> IO a
            withFile name mode f = bracket (openFile name mode)
                (\handle -> hClose handle)
                (\handle -> f handle)
            ```
        - Other common IO functions
            - `readFile :: FilePath -> IO String`
            - `writeFile :: FilePath -> String -> IO ()`
            - `appendFile :: FilePath -> String -> IO ()`
    - *9.3 - To-Do Lists*
    - *9.4 - Command-Line Arguments*

        ```haskell
        import System.Environment

        main = do
            args <- getArgs
            progName <- getProgName
            putStrLn "The arguments are:"
            mapM putStrLn args
            putStrLn "The program name is:"
            putStrLn progName
        ```


















