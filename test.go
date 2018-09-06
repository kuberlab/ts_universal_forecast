package main

import (
	"bytes"
	"fmt"
	"github.com/Masterminds/sprig"
	"io/ioutil"
	"os"
	"text/template"
)

func FuncMap() template.FuncMap {
	f := sprig.TxtFuncMap()
	delete(f, "env")
	delete(f, "expandenv")
	return f
}

func main() {
	t := template.New("gotpl")
	t = t.Funcs(FuncMap())
	f, err := os.Open("task-template.yaml")
	if err != nil {
		panic(err)
	}
	data, err := ioutil.ReadAll(f)
	if err != nil {
		panic(err)
	}
	t, err = t.Parse(string(data))
	if err != nil {
		panic(err)
	}
	buffer := bytes.NewBuffer(make([]byte, 0))

	vars := map[string]interface{}{
		"OutputWidowLength": 10,
		"InputWindowLength":  100,
		"TrainingFiles":     "test",
		"EvalutionFiles":    "test",
		"ExogenousColumns": []string{"c1","c2","c3"},
		"ExcludeColumns": []string{"c4","c5","c6"},
		"TimestampColumn": "time",
		"TimestampColumnFormat": "unix",
	}
	if err := t.ExecuteTemplate(buffer, "gotpl", vars); err != nil {
		panic(err)
	}
	fmt.Println(buffer.String())
}
