# tp5-sia


## Para instalar dependencias
```
> pip install -r requirements.txt
```

## CONFIGURACION EJERCICIO 1b

seed: numero positivo, es la semilla para la generacion de valores aleatorios
layer_config: Define la arquitectura de red, pasar un arreglo que determine cantidad
de neuronas en capas ocultas y superficiales, la primera capa y la ultima deben tener 35 para que matcheen con los inputs y se recomienda poner 2 para el espacio latente.
ej: [35,23,2,23,35]

Optimizer: propiedad type: adam o momentum
    si se elige adam aclarar
        a) alpha
        b) beta1
        c) beta2
        d) epsilon
    si se elige momentum aclarar
        a) learning_rate

activation_function_beta -> parametro beta de las funciones sigmodeas

batch_size: Tamanio de batch de entrenamiento, dejar en -1 para que se use todo el conjunto en un batch

epochs: Cantidad de epocas para entrenar.
strategy

1-Si se opta por: "bit_fliping_with_n"
    Aclarar
        a) n_min: minimo numero de bits a invertirse en posiciones aleatorias
        b) n_max: maximo numero de bits a invertirse en posiciones aleatorias
        se avanza entre n_min y n_max con paso 1
        
2- Si se opta por: "bit_flip_with_probability"
    Aclarar
        a) probability_min: Probabilidad minima para invertir un bit en una posicion determinada
        b) probability_max: Probabilidad maxima para invertir un bit en una posicion determinada
        c) step: Aumento de probabilidad en cada iteracion
