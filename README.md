# Trabajo Práctico 5 - SIA

## Requisitos:
1) Python 3
2) Pip 3


## Para instalar dependencias
```
> pip install -r requirements.txt
```

## Configuración General

| Campo                     | Descripción                                                                                                                                                                                                                                       | Valores aceptados                                                            |  
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| seed                      | Permite especificar el numero inicial usado al generar los valores aleatorios.                                                                                                                                                                    | Debe entero positivo. Tambien se puede usar -1 si no se quiere usar un seed. |
| layer_config              | Define la arquitectura de red, pasar un arreglo que determine cantidad de neuronas en capas ocultas y superficiales, la primera capa y la ultima deben tener 35 para que matcheen con los inputs y se recomienda poner 2 para el espacio latente. | Arreglo de Int.                                                              |
| optimizer.type            | Adam o Momentum.                                                                                                                                                                                                                                  | Numero de punto flotante entre 0 y 1.                                        |
| optimizer.alpha           | Valor de alpha. Poner en el caso que se use ADAM.                                                                                                                                                                                                 | Numero de punto flotante entre 0 y 1.                                        | 
| optimizer.beta1           | Valor de beta1. Poner en el caso que se use ADAM.                                                                                                                                                                                                 | Numero de punto flotante entre 0 y 1.                                        | 
| optimizer.beta2           | Valor de beta2. Poner en el caso que se use ADAM.                                                                                                                                                                                                 | Numero de punto flotante entre 0 y 1.                                        | 
| optimizer.epsilon         | Valor de epsilon. Poner en el caso que se use ADAM.                                                                                                                                                                                               | Numero de punto flotante entre 0 y 1.                                        | 
| optimizer.learning_rate   | Valor del learning rate. Poner en el caso que se use Momentum.                                                                                                                                                                                    | Numero de punto flotante entre 0 y 1.                                        |
| epochs                    | Maxima cantidad de iteraciones al entrenar.                                                                                                                                                                                                       | Numero entero mayor a 0.                                                     | 
| activation_function_beta  | Valor de beta en funcion sigmoid.                                                                                                                                                                                                                 | Numero flotante positivo.                                                    | 

## Configuración Ejercicio 1b
- strategy

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
